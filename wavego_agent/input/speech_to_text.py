"""
Speech-to-Text Module
=====================
Handles voice input and converts speech to text using various engines.
Supports Google Speech Recognition, OpenAI Whisper (API and local), and Vosk.
"""

import os
import time
import threading
import queue
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any


class STTEngine(ABC):
    """Abstract base class for Speech-to-Text engines."""
    
    @abstractmethod
    def transcribe(self, audio_data) -> Optional[str]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data in appropriate format for the engine
            
        Returns:
            Transcribed text or None if failed
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine is available and properly configured."""
        pass


class GoogleSTT(STTEngine):
    """Google Speech Recognition engine (requires internet)."""
    
    def __init__(self, language: str = "en-US"):
        """
        Initialize Google STT.
        
        Args:
            language: Language code for recognition
        """
        self.language = language
        self._recognizer = None
        self._init_recognizer()
    
    def _init_recognizer(self):
        """Initialize the speech recognizer."""
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
        except ImportError:
            print("[GoogleSTT] speech_recognition not installed. "
                  "Install with: pip install SpeechRecognition")
    
    def is_available(self) -> bool:
        """Check if Google STT is available."""
        return self._recognizer is not None
    
    def transcribe(self, audio_data) -> Optional[str]:
        """Transcribe using Google Speech Recognition."""
        if not self.is_available():
            return None
        
        try:
            import speech_recognition as sr
            text = self._recognizer.recognize_google(
                audio_data, 
                language=self.language
            )
            return text
        except Exception as e:
            print(f"[GoogleSTT] Recognition failed: {e}")
            return None


class WhisperAPISTT(STTEngine):
    """OpenAI Whisper API engine."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Whisper API STT.
        
        Args:
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client."""
        if not self.api_key:
            print("[WhisperAPI] No API key provided")
            return
        
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            print("[WhisperAPI] openai not installed. "
                  "Install with: pip install openai")
    
    def is_available(self) -> bool:
        """Check if Whisper API is available."""
        return self._client is not None
    
    def transcribe(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe using OpenAI Whisper API.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text or None
        """
        if not self.is_available():
            return None
        
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcription = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcription.text
        except Exception as e:
            print(f"[WhisperAPI] Transcription failed: {e}")
            return None


class WhisperLocalSTT(STTEngine):
    """Local Whisper model engine (runs on device)."""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize local Whisper STT.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self._model = None
        self._init_model()
    
    def _init_model(self):
        """Load the Whisper model."""
        try:
            import whisper
            print(f"[WhisperLocal] Loading model '{self.model_name}'...")
            self._model = whisper.load_model(self.model_name)
            print("[WhisperLocal] Model loaded successfully")
        except ImportError:
            print("[WhisperLocal] whisper not installed. "
                  "Install with: pip install openai-whisper")
        except Exception as e:
            print(f"[WhisperLocal] Failed to load model: {e}")
    
    def is_available(self) -> bool:
        """Check if local Whisper is available."""
        return self._model is not None
    
    def transcribe(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe using local Whisper model.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text or None
        """
        if not self.is_available():
            return None
        
        try:
            result = self._model.transcribe(audio_file_path)
            return result["text"].strip()
        except Exception as e:
            print(f"[WhisperLocal] Transcription failed: {e}")
            return None


class VoskSTT(STTEngine):
    """Vosk offline speech recognition engine."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Vosk STT.
        
        Args:
            model_path: Path to Vosk model directory
        """
        self.model_path = model_path or os.environ.get("VOSK_MODEL_PATH")
        self._model = None
        self._init_model()
    
    def _init_model(self):
        """Load the Vosk model."""
        if not self.model_path:
            print("[Vosk] No model path provided. Set VOSK_MODEL_PATH env var "
                  "or pass model_path parameter")
            return
        
        try:
            from vosk import Model
            self._model = Model(self.model_path)
        except ImportError:
            print("[Vosk] vosk not installed. Install with: pip install vosk")
        except Exception as e:
            print(f"[Vosk] Failed to load model: {e}")
    
    def is_available(self) -> bool:
        """Check if Vosk is available."""
        return self._model is not None
    
    def transcribe(self, audio_data) -> Optional[str]:
        """Transcribe using Vosk."""
        if not self.is_available():
            return None
        
        try:
            from vosk import KaldiRecognizer
            import json
            
            rec = KaldiRecognizer(self._model, 16000)
            rec.AcceptWaveform(audio_data)
            result = json.loads(rec.FinalResult())
            return result.get("text", "")
        except Exception as e:
            print(f"[Vosk] Transcription failed: {e}")
            return None


class SpeechToText:
    """
    Main Speech-to-Text interface.
    
    Provides a unified interface for different STT engines with support for
    continuous listening, wake word detection, and callbacks.
    
    Example:
        stt = SpeechToText(engine="google", language="en-US")
        
        # Single recognition
        text = stt.listen_once()
        print(f"You said: {text}")
        
        # Continuous listening with callback
        def on_speech(text):
            print(f"Detected: {text}")
        
        stt.start_continuous(callback=on_speech)
    """
    
    def __init__(
        self,
        engine: str = "google",
        language: str = "en-US",
        wake_word: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Speech-to-Text module.
        
        Args:
            engine: STT engine to use ("google", "whisper_api", "whisper_local", "vosk")
            language: Language code for recognition
            wake_word: Optional wake word to trigger recognition
            config: Additional configuration dictionary
        """
        self.engine_name = engine
        self.language = language
        self.wake_word = wake_word.lower() if wake_word else None
        self.config = config or {}
        
        self._engine: Optional[STTEngine] = None
        self._recognizer = None
        self._microphone = None
        self._is_listening = False
        self._listen_thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._callback: Optional[Callable[[str], None]] = None
        
        self._init_audio()
        self._init_engine()
    
    def _init_audio(self):
        """Initialize audio input."""
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            
            # Configure recognizer
            self._recognizer.energy_threshold = self.config.get("energy_threshold", 300)
            self._recognizer.dynamic_energy_threshold = self.config.get("dynamic_energy", True)
            self._recognizer.pause_threshold = self.config.get("pause_threshold", 0.8)
            
            # Get microphone
            device_index = self.config.get("mic_device_index")
            self._microphone = sr.Microphone(device_index=device_index)
            
            # Calibrate for ambient noise
            print("[STT] Calibrating for ambient noise...")
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
            print("[STT] Calibration complete")
            
        except ImportError:
            print("[STT] speech_recognition not installed. "
                  "Install with: pip install SpeechRecognition pyaudio")
        except Exception as e:
            print(f"[STT] Audio initialization failed: {e}")
    
    def _init_engine(self):
        """Initialize the selected STT engine."""
        if self.engine_name == "google":
            self._engine = GoogleSTT(language=self.language)
        elif self.engine_name == "whisper_api":
            api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
            self._engine = WhisperAPISTT(api_key=api_key)
        elif self.engine_name == "whisper_local":
            model = self.config.get("whisper_model", "base")
            self._engine = WhisperLocalSTT(model_name=model)
        elif self.engine_name == "vosk":
            model_path = self.config.get("vosk_model_path")
            self._engine = VoskSTT(model_path=model_path)
        else:
            print(f"[STT] Unknown engine: {self.engine_name}, defaulting to Google")
            self._engine = GoogleSTT(language=self.language)
    
    def is_available(self) -> bool:
        """Check if STT is available and ready."""
        return (
            self._recognizer is not None and 
            self._microphone is not None and
            self._engine is not None and
            self._engine.is_available()
        )
    
    def listen_once(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> Optional[str]:
        """
        Listen for a single utterance and return transcribed text.
        
        Args:
            timeout: Max seconds to wait for speech to start
            phrase_time_limit: Max seconds for the utterance
            
        Returns:
            Transcribed text or None if failed
        """
        if not self.is_available():
            print("[STT] Not available")
            return None
        
        try:
            import speech_recognition as sr
            
            print("[STT] Listening...")
            with self._microphone as source:
                audio = self._recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            print("[STT] Processing...")
            
            # For Google STT, use recognizer directly
            if isinstance(self._engine, GoogleSTT):
                text = self._engine.transcribe(audio)
            else:
                # For other engines, save to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio.get_wav_data())
                    temp_path = f.name
                
                text = self._engine.transcribe(temp_path)
                os.unlink(temp_path)
            
            if text:
                print(f"[STT] Recognized: {text}")
                
                # Check wake word if configured
                if self.wake_word:
                    if self.wake_word in text.lower():
                        # Remove wake word from text
                        text = text.lower().replace(self.wake_word, "").strip()
                        return text if text else None
                    else:
                        return None
                
                return text
            
            return None
            
        except Exception as e:
            print(f"[STT] Listen failed: {e}")
            return None
    
    def start_continuous(self, callback: Callable[[str], None]):
        """
        Start continuous listening in background.
        
        Args:
            callback: Function to call with each recognized text
        """
        if self._is_listening:
            print("[STT] Already listening")
            return
        
        self._callback = callback
        self._is_listening = True
        self._listen_thread = threading.Thread(target=self._continuous_listen_loop, daemon=True)
        self._listen_thread.start()
        print("[STT] Started continuous listening")
    
    def stop_continuous(self):
        """Stop continuous listening."""
        self._is_listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
            self._listen_thread = None
        print("[STT] Stopped continuous listening")
    
    def _continuous_listen_loop(self):
        """Background loop for continuous listening."""
        while self._is_listening:
            try:
                text = self.listen_once(timeout=3.0, phrase_time_limit=10.0)
                if text and self._callback:
                    self._callback(text)
            except Exception as e:
                print(f"[STT] Continuous listen error: {e}")
                time.sleep(0.5)


# Factory function
def create_stt(config: Dict[str, Any]) -> SpeechToText:
    """
    Create a SpeechToText instance from configuration.
    
    Args:
        config: Configuration dictionary with speech settings
        
    Returns:
        Configured SpeechToText instance
    """
    speech_config = config.get("speech", {})
    audio_config = config.get("hardware", {}).get("audio", {})
    
    return SpeechToText(
        engine=speech_config.get("stt_engine", "google"),
        language=speech_config.get("language", "en-US"),
        wake_word=speech_config.get("wake_word"),
        config={
            "mic_device_index": audio_config.get("mic_device_index"),
            "api_key": os.environ.get(speech_config.get("whisper", {}).get("api_key_env", "OPENAI_API_KEY")),
            "whisper_model": speech_config.get("whisper", {}).get("model", "base"),
        }
    )
