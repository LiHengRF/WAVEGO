#!/usr/bin/env python3
"""
WaveGo Agent - Main Entry Point
================================
Voice + Vision controlled robot dog using LLM for intelligent decision making.

Usage:
    python main.py                    # Run with default config
    python main.py --config my.yaml   # Run with custom config
    python main.py --mock             # Run in mock mode (no hardware)
    python main.py --test             # Run interactive test mode

Architecture:
    Input Layer:  Speech-to-Text + Vision Processing
    Brain Layer:  LLM API + Local Agent Decision Making
    Output Layer: Motor Controller + Text-to-Speech
"""

import os
import sys
import argparse
import signal
import time
import yaml
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core import get_state_manager
from input import create_stt, create_vision_processor
from brain import create_llm_client, create_agent, AgentLoop
from output import create_motor_controller, create_tts

# Web dashboard (optional)
try:
    from web import create_dashboard_server, dashboard, run_standalone as run_web_standalone
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False


class WaveGoAgent:
    """
    Main WaveGo Agent application.
    
    Coordinates all components:
    - Speech-to-Text (voice input)
    - Vision Processor (camera/OpenCV)
    - LLM Client (language understanding)
    - Agent (decision making)
    - Motor Controller (ESP32 communication)
    - Text-to-Speech (voice output)
    
    Example:
        agent = WaveGoAgent(config_path="config/config.yaml")
        agent.start()
        
        # Interactive mode
        agent.process_command("Move forward half a meter")
        
        # Or continuous listening
        agent.start_listening()
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        use_mock: bool = False
    ):
        """
        Initialize the WaveGo Agent.
        
        Args:
            config_path: Path to configuration YAML file
            use_mock: If True, use mock components for testing
        """
        self.use_mock = use_mock
        self.config = self._load_config(config_path)
        
        # Components
        self.state_manager = get_state_manager()
        self.stt = None
        self.vision = None
        self.llm = None
        self.agent = None
        self.motor = None
        self.tts = None
        self.agent_loop = None
        
        # State
        self._is_running = False
        
        # Initialize components
        self._init_components()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                print(f"[WaveGo] Loaded config from: {config_path}")
                return config
        else:
            print(f"[WaveGo] Config not found at {config_path}, using defaults")
            return {}
    
    def _init_components(self):
        """Initialize all system components."""
        print("[WaveGo] Initializing components...")
        
        # Speech-to-Text
        try:
            self.stt = create_stt(self.config)
            print(f"[WaveGo] STT: {'available' if self.stt.is_available() else 'not available'}")
        except Exception as e:
            print(f"[WaveGo] STT init failed: {e}")
        
        # Vision Processor
        try:
            self.vision = create_vision_processor(self.config)
            print("[WaveGo] Vision processor created")
        except Exception as e:
            print(f"[WaveGo] Vision init failed: {e}")
        
        # LLM Client
        try:
            self.llm = create_llm_client(self.config, use_mock=self.use_mock)
            print(f"[WaveGo] LLM: {'available' if self.llm.is_available() else 'not available (using mock)'}")
            if not self.llm.is_available() and not self.use_mock:
                print("[WaveGo] Falling back to mock LLM")
                from brain import MockLLMClient
                self.llm = MockLLMClient()
        except Exception as e:
            print(f"[WaveGo] LLM init failed: {e}")
            from brain import MockLLMClient
            self.llm = MockLLMClient()
        
        # Motor Controller
        try:
            self.motor = create_motor_controller(self.config, use_mock=self.use_mock)
            if not self.use_mock:
                self.motor.connect()
            print(f"[WaveGo] Motor: {'connected' if self.motor.is_connected() else 'mock/not connected'}")
        except Exception as e:
            print(f"[WaveGo] Motor init failed: {e}")
        
        # Text-to-Speech
        try:
            self.tts = create_tts(self.config, use_mock=self.use_mock)
            print(f"[WaveGo] TTS: {'available' if self.tts.is_available() else 'not available'}")
        except Exception as e:
            print(f"[WaveGo] TTS init failed: {e}")
        
        # Agent
        if self.llm:
            self.agent = create_agent(self.llm, self.config)
            print("[WaveGo] Agent created")
        
        print("[WaveGo] Initialization complete")
    
    def start(self):
        """Start the agent system (vision processing)."""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start vision processing
        if self.vision:
            self.vision.start()
            # Register vision state callback
            self.vision.set_callback(self._on_vision_update)
        
        print("[WaveGo] Agent started")
        
        # Speak startup message
        if self.tts:
            self.tts.speak("WaveGo robot ready.")
    
    def stop(self):
        """Stop the agent system."""
        self._is_running = False
        
        # Stop continuous listening
        if self.stt:
            self.stt.stop_continuous()
        
        # Stop vision
        if self.vision:
            self.vision.stop()
        
        # Stop motors
        if self.motor:
            self.motor.stop()
            self.motor.disconnect()
        
        # Stop TTS
        if self.tts:
            self.tts.shutdown()
        
        print("[WaveGo] Agent stopped")
    
    def start_listening(self):
        """Start continuous voice command listening."""
        if not self.stt:
            print("[WaveGo] STT not available")
            return
        
        print("[WaveGo] Starting continuous listening...")
        self.stt.start_continuous(callback=self._on_speech)
    
    def stop_listening(self):
        """Stop continuous listening."""
        if self.stt:
            self.stt.stop_continuous()
    
    def _on_speech(self, text: str):
        """Callback for speech recognition."""
        if not text:
            return
        
        print(f"\n[WaveGo] Heard: '{text}'")
        self.process_command(text)
    
    def _on_vision_update(self, state):
        """Callback for vision state updates."""
        # Update state manager
        self.state_manager.update_vision(state)
    
    def process_command(self, command: str) -> Optional[str]:
        """
        Process a single command.
        
        Args:
            command: Text command to process
            
        Returns:
            Agent's response text
        """
        if not command or not self.agent:
            return None
        
        # Get current vision state
        vision_state = None
        if self.vision:
            vision_state = self.vision.get_vision_state()
        
        # Make decision
        decision = self.agent.decide(command, vision_state)
        
        print(f"[WaveGo] Intent: {decision.intent.value}")
        print(f"[WaveGo] Commands: {len(decision.motor_commands)}")
        print(f"[WaveGo] Reply: {decision.reply_text}")
        
        # Execute motor commands
        if self.motor and decision.motor_commands:
            for cmd in decision.motor_commands:
                self.motor.execute(cmd)
                if cmd.duration:
                    time.sleep(cmd.duration)
        
        # Speak response
        if self.tts and decision.reply_text:
            self.tts.speak(decision.reply_text)
        
        # Update state
        if decision.motor_commands:
            self.state_manager.update_action(
                f"{decision.intent.value}: {decision.motor_commands[0].command_type}"
            )
        
        return decision.reply_text
    
    def interactive_mode(self):
        """
        Run in interactive text mode.
        
        Useful for testing without voice input.
        """
        print("\n" + "="*60)
        print("WaveGo Agent - Interactive Mode")
        print("="*60)
        print("Type commands to control the robot.")
        print("Commands: 'quit' to exit, 'status' for robot status")
        print("="*60 + "\n")
        
        while True:
            try:
                command = input("You: ").strip()
                
                if not command:
                    continue
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if command.lower() == 'status':
                    if self.agent:
                        print(f"Robot: {self.agent.get_status_response()}")
                    continue
                
                if command.lower() == 'vision':
                    if self.vision:
                        state = self.vision.get_vision_state()
                        print(f"Vision: {state.to_dict()}")
                    continue
                
                response = self.process_command(command)
                if response:
                    print(f"Robot: {response}\n")
                
            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WaveGo Agent - Voice + Vision controlled robot dog"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Run in mock mode (no hardware)"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run in interactive test mode"
    )
    parser.add_argument(
        "--voice", "-v",
        action="store_true",
        help="Enable voice control (continuous listening)"
    )
    parser.add_argument(
        "--web", "-w",
        action="store_true",
        help="Enable web dashboard interface"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Web dashboard port (default: 5000)"
    )
    
    args = parser.parse_args()
    
    # Create agent
    agent = WaveGoAgent(
        config_path=args.config,
        use_mock=args.mock
    )
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        print("\n[WaveGo] Shutting down...")
        agent.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start agent
    agent.start()
    
    # Setup web dashboard if requested
    if args.web:
        if not WEB_AVAILABLE:
            print("[WaveGo] Web module not available. Install Flask: pip install flask flask-cors")
            sys.exit(1)
        
        # Connect dashboard to agent components
        dashboard.set_components(
            agent=agent.agent,
            motor=agent.motor,
            vision=agent.vision,
            stt=agent.stt,
            tts=agent.tts,
            state_manager=agent.state_manager
        )
        
        # Setup vision frame callback for video streaming
        if agent.vision:
            original_callback = agent.vision._callback
            def combined_callback(state):
                if original_callback:
                    original_callback(state)
                # Push frame to dashboard
                frame = agent.vision.get_frame()
                if frame is not None:
                    dashboard.push_frame(frame)
                dashboard.update_vision_state(state.to_dict())
            agent.vision.set_callback(combined_callback)
        
        print(f"[WaveGo] Web dashboard available at http://0.0.0.0:{args.port}")
    
    try:
        if args.web:
            # Web dashboard mode
            from web import app
            import threading
            
            if args.voice:
                # Also start voice control in background
                agent.start_listening()
                print("[WaveGo] Voice control active alongside web dashboard")
            
            # Run Flask in main thread
            print(f"[WaveGo] Starting web server on port {args.port}...")
            app.run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)
        elif args.test or args.mock:
            # Interactive mode
            agent.interactive_mode()
        elif args.voice:
            # Voice control mode
            agent.start_listening()
            print("[WaveGo] Voice control active. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        else:
            # Default: interactive mode
            agent.interactive_mode()
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
