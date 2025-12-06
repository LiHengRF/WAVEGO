"""
Text-to-Speech Module
=====================
Handles converting text responses to audio output.
Supports multiple TTS engines: pyttsx3 (offline), gTTS, and OpenAI TTS.
"""

import os
import time
import threading
import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from queue import Queue


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""
    
    @abstractmethod
    def speak(self, text: str) -> bool:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine is available."""
        pass
    
    def stop(self):
        """Stop current speech (if supported)."""
        pass


class Pyttsx3Engine(TTSEngine):
    """
    Offline TTS engine using pyttsx3.
    
    Works completely offline, good for Raspberry Pi.
    """
    
    def __init__(self, rate: int = 150, volume: float = 0.9):
        """
        Initialize pyttsx3 engine.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume (0.0 to 1.0)
        """
        self.rate = rate
        self.volume = volume
        self._engine = None
        self._lock = threading.Lock()
        self._init_engine()
    
    def _init_engine(self):
        """Initialize the pyttsx3 engine."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
            print("[TTS-pyttsx3] Initialized")
        except ImportError:
            print("[TTS-pyttsx3] pyttsx3 not installed. "
                  "Install with: pip install pyttsx3")
        except Exception as e:
            print(f"[TTS-pyttsx3] Failed to initialize: {e}")
    
    def is_available(self) -> bool:
        return self._engine is not None
    
    def speak(self, text: str) -> bool:
        if not self.is_available():
            return False
        
        try:
            with self._lock:
                self._engine.say(text)
                self._engine.runAndWait()
            return True
        except Exception as e:
            print(f"[TTS-pyttsx3] Error: {e}")
            return False
    
    def stop(self):
        if self._engine:
            try:
                self._engine.stop()
            except:
                pass


class GTTSEngine(TTSEngine):
    """
    Google TTS engine (requires internet).
    
    Higher quality than pyttsx3 but requires network.
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize gTTS engine.
        
        Args:
            language: Language code
        """
        self.language = language
        self._available = False
        self._check_available()
    
    def _check_available(self):
        """Check if gTTS is available."""
        try:
            from gtts import gTTS
            import pygame
            pygame.mixer.init()
            self._available = True
            print("[TTS-gTTS] Initialized")
        except ImportError as e:
            print(f"[TTS-gTTS] Missing dependency: {e}")
            print("Install with: pip install gtts pygame")
    
    def is_available(self) -> bool:
        return self._available
    
    def speak(self, text: str) -> bool:
        if not self.is_available():
            return False
        
        try:
            from gtts import gTTS
            import pygame
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name
            
            # Generate speech
            tts = gTTS(text=text, lang=self.language)
            tts.save(temp_path)
            
            # Play audio
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up
            os.unlink(temp_path)
            
            return True
        except Exception as e:
            print(f"[TTS-gTTS] Error: {e}")
            return False
    
    def stop(self):
        try:
            import pygame
            pygame.mixer.music.stop()
        except:
            pass


class OpenAITTSEngine(TTSEngine):
    """
    OpenAI TTS engine (high quality, requires API key).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "tts-1",
        voice: str = "alloy"
    ):
        """
        Initialize OpenAI TTS engine.
        
        Args:
            api_key: OpenAI API key
            model: TTS model (tts-1 or tts-1-hd)
            voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.voice = voice
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client."""
        if not self.api_key:
            print("[TTS-OpenAI] No API key provided")
            return
        
        try:
            from openai import OpenAI
            import pygame
            pygame.mixer.init()
            self._client = OpenAI(api_key=self.api_key)
            print(f"[TTS-OpenAI] Initialized with voice: {self.voice}")
        except ImportError as e:
            print(f"[TTS-OpenAI] Missing dependency: {e}")
    
    def is_available(self) -> bool:
        return self._client is not None
    
    def speak(self, text: str) -> bool:
        if not self.is_available():
            return False
        
        try:
            import pygame
            
            # Generate speech
            response = self._client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text
            )
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name
                response.stream_to_file(temp_path)
            
            # Play audio
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            os.unlink(temp_path)
            return True
            
        except Exception as e:
            print(f"[TTS-OpenAI] Error: {e}")
            return False
    
    def stop(self):
        try:
            import pygame
            pygame.mixer.music.stop()
        except:
            pass


class TextToSpeech:
    """
    Main Text-to-Speech interface.
    
    Provides a unified interface for different TTS engines with
    async speech queue support.
    
    Example:
        tts = TextToSpeech(engine="pyttsx3")
        tts.speak("Hello, I am your robot assistant.")
        
        # Async mode
        tts.speak_async("Processing your command...")
    """
    
    def __init__(
        self,
        engine: str = "pyttsx3",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Text-to-Speech.
        
        Args:
            engine: Engine to use ("pyttsx3", "gtts", "openai")
            config: Engine-specific configuration
        """
        self.engine_name = engine
        self.config = config or {}
        
        self._engine: Optional[TTSEngine] = None
        self._speech_queue: Queue = Queue()
        self._queue_thread: Optional[threading.Thread] = None
        self._is_running = False
        
        self._init_engine()
    
    def _init_engine(self):
        """Initialize the selected TTS engine."""
        if self.engine_name == "pyttsx3":
            self._engine = Pyttsx3Engine(
                rate=self.config.get("voice_rate", 150),
                volume=self.config.get("voice_volume", 0.9)
            )
        elif self.engine_name == "gtts":
            self._engine = GTTSEngine(
                language=self.config.get("language", "en")
            )
        elif self.engine_name == "openai":
            openai_config = self.config.get("openai_tts", {})
            self._engine = OpenAITTSEngine(
                api_key=os.environ.get(self.config.get("api_key_env", "OPENAI_API_KEY")),
                model=openai_config.get("model", "tts-1"),
                voice=openai_config.get("voice", "alloy")
            )
        else:
            print(f"[TTS] Unknown engine: {self.engine_name}, using pyttsx3")
            self._engine = Pyttsx3Engine()
        
        # Start queue processing thread
        self._start_queue_thread()
    
    def _start_queue_thread(self):
        """Start the speech queue processing thread."""
        self._is_running = True
        self._queue_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._queue_thread.start()
    
    def _process_queue(self):
        """Process speech queue in background."""
        while self._is_running:
            try:
                text = self._speech_queue.get(timeout=0.5)
                if text and self._engine:
                    self._engine.speak(text)
            except:
                pass
    
    def is_available(self) -> bool:
        """Check if TTS is available."""
        return self._engine is not None and self._engine.is_available()
    
    def speak(self, text: str) -> bool:
        """
        Speak text (blocking).
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful
        """
        if not text:
            return True
        
        if not self.is_available():
            print(f"[TTS] Not available, would say: {text}")
            return False
        
        print(f"[TTS] Speaking: {text}")
        return self._engine.speak(text)
    
    def speak_async(self, text: str):
        """
        Queue text for async speech (non-blocking).
        
        Args:
            text: Text to speak
        """
        if text:
            self._speech_queue.put(text)
    
    def stop(self):
        """Stop current speech and clear queue."""
        # Clear queue
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except:
                pass
        
        # Stop current speech
        if self._engine:
            self._engine.stop()
    
    def shutdown(self):
        """Shutdown TTS system."""
        self._is_running = False
        self.stop()
        if self._queue_thread:
            self._queue_thread.join(timeout=2.0)


class MockTTS(TextToSpeech):
    """Mock TTS for testing - just prints text."""
    
    def __init__(self, **kwargs):
        self._speech_queue = Queue()
        self._is_running = False
    
    def is_available(self) -> bool:
        return True
    
    def speak(self, text: str) -> bool:
        print(f"[MockTTS] Would speak: {text}")
        return True
    
    def speak_async(self, text: str):
        print(f"[MockTTS] Would speak (async): {text}")
    
    def stop(self):
        pass
    
    def shutdown(self):
        pass


# Factory function
def create_tts(config: Dict[str, Any], use_mock: bool = False) -> TextToSpeech:
    """
    Create a TextToSpeech instance from configuration.
    
    Args:
        config: Configuration dictionary
        use_mock: If True, create a mock TTS
        
    Returns:
        Configured TextToSpeech instance
    """
    if use_mock:
        return MockTTS()
    
    tts_config = config.get("tts", {})
    
    return TextToSpeech(
        engine=tts_config.get("engine", "pyttsx3"),
        config=tts_config
    )
