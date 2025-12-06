#!/usr/bin/env python3
"""
WaveGo Agent - Examples and Demos
=================================
This file demonstrates how to use each component of the WaveGo Agent system.
Run individual examples to test specific functionality.

Usage:
    python examples.py motor    # Test motor controller
    python examples.py vision   # Test vision processor
    python examples.py stt      # Test speech-to-text
    python examples.py tts      # Test text-to-speech
    python examples.py llm      # Test LLM client
    python examples.py agent    # Test agent decision making
    python examples.py all      # Run all examples
"""

import sys
import os
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def example_motor_controller():
    """
    Example: Motor Controller
    
    Demonstrates how to send commands to the ESP32.
    """
    print("\n" + "="*60)
    print("Example: Motor Controller")
    print("="*60)
    
    from output.motor_controller import MotorController, MockMotorController
    
    # Use mock controller for demo (no hardware needed)
    motor = MockMotorController()
    motor.connect()
    
    print("\n1. Basic movement commands:")
    motor.forward(duration=1.0)
    time.sleep(0.2)
    motor.backward(duration=0.5)
    time.sleep(0.2)
    motor.left(duration=0.3)
    time.sleep(0.2)
    motor.right(duration=0.3)
    time.sleep(0.2)
    motor.stop()
    
    print("\n2. Head/look commands:")
    motor.look_up()
    motor.look_down()
    motor.look_left()
    motor.look_right()
    motor.stop_head()
    
    print("\n3. Gesture commands:")
    motor.jump()
    motor.handshake()
    motor.steady_mode()
    
    print("\n4. Light control:")
    motor.set_light("blue")
    motor.set_light("red")
    motor.set_light("off")
    
    print("\n5. Using MotorCommand objects:")
    from core import MotorCommand
    cmd = MotorCommand(command_type="move", value=1, duration=1.5)
    motor.execute(cmd)
    
    motor.disconnect()
    print("\nMotor controller example complete!")


def example_vision_processor():
    """
    Example: Vision Processor
    
    Demonstrates camera processing and obstacle detection.
    """
    print("\n" + "="*60)
    print("Example: Vision Processor")
    print("="*60)
    
    from input.vision_processor import VisionProcessor
    from core import VisionState
    
    # Create processor (will use camera if available)
    print("\nCreating vision processor...")
    processor = VisionProcessor(camera_id=0, width=640, height=480)
    
    # Try to start (may fail without camera)
    try:
        if processor.start():
            print("Vision processor started!")
            
            # Get vision state a few times
            for i in range(3):
                time.sleep(0.5)
                state = processor.get_vision_state()
                print(f"\nVision state {i+1}:")
                print(f"  Obstacle ahead: {state.obstacle_ahead}")
                print(f"  Face detected: {state.face_detected} (count: {state.face_count})")
                print(f"  Color target: {state.color_target_detected}")
                print(f"  Motion detected: {state.motion_detected}")
            
            processor.stop()
        else:
            print("Could not start vision processor (no camera?)")
    except Exception as e:
        print(f"Vision processor error: {e}")
    
    # Demonstrate VisionState structure
    print("\nVisionState structure example:")
    state = VisionState(
        obstacle_ahead=True,
        obstacle_distance=0.5,
        face_detected=True,
        face_count=2,
        face_positions=[(100, 100, 50, 50), (200, 150, 60, 60)],
        color_target_detected=False,
        motion_detected=True,
        motion_area=(300, 200, 100, 100)
    )
    print(f"  to_dict(): {state.to_dict()}")
    
    print("\nVision processor example complete!")


def example_speech_to_text():
    """
    Example: Speech-to-Text
    
    Demonstrates voice recognition.
    """
    print("\n" + "="*60)
    print("Example: Speech-to-Text")
    print("="*60)
    
    from input.speech_to_text import SpeechToText
    
    print("\nCreating STT with Google engine...")
    stt = SpeechToText(engine="google", language="en-US")
    
    if stt.is_available():
        print("STT is available!")
        print("\nTry speaking a command (you have 5 seconds)...")
        
        text = stt.listen_once(timeout=5.0, phrase_time_limit=10.0)
        
        if text:
            print(f"Recognized: '{text}'")
        else:
            print("No speech detected or recognition failed")
    else:
        print("STT not available (missing pyaudio or microphone?)")
    
    print("\nSTT example complete!")


def example_text_to_speech():
    """
    Example: Text-to-Speech
    
    Demonstrates voice output.
    """
    print("\n" + "="*60)
    print("Example: Text-to-Speech")
    print("="*60)
    
    from output.text_to_speech import TextToSpeech, MockTTS
    
    # Try real TTS
    print("\nTrying pyttsx3 engine...")
    tts = TextToSpeech(engine="pyttsx3")
    
    if tts.is_available():
        print("TTS is available!")
        tts.speak("Hello! I am your WaveGo robot assistant.")
        tts.speak("I can move, look around, and respond to your commands.")
    else:
        print("TTS not available, using mock...")
        tts = MockTTS()
        tts.speak("Hello! I am your WaveGo robot assistant.")
    
    tts.shutdown()
    print("\nTTS example complete!")


def example_llm_client():
    """
    Example: LLM Client
    
    Demonstrates language model integration.
    """
    print("\n" + "="*60)
    print("Example: LLM Client")
    print("="*60)
    
    from brain.llm_client import LLMClient, MockLLMClient
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        print("\nUsing real OpenAI API...")
        llm = LLMClient(api_key=api_key, model="gpt-4o-mini")
    else:
        print("\nNo API key found, using mock LLM...")
        llm = MockLLMClient()
    
    # Test commands
    test_commands = [
        "Move forward half a meter",
        "Turn left",
        "What's in front of me?",
        "Jump!",
        "Turn on the blue lights",
    ]
    
    # Context for LLM
    context = {
        "robot_state": {"mode": "idle", "is_moving": False},
        "vision_state": {"obstacle_ahead": False, "face_detected": False}
    }
    
    for cmd in test_commands:
        print(f"\n> User: {cmd}")
        response = llm.chat(cmd, context=context)
        
        if response.success and response.parsed:
            print(f"  Intent: {response.parsed.get('intent')}")
            print(f"  Action: {response.parsed.get('action')}")
            print(f"  Reply: {response.parsed.get('reply')}")
        else:
            print(f"  Error: {response.error}")
    
    print("\nLLM client example complete!")


def example_agent():
    """
    Example: Agent Decision Making
    
    Demonstrates the full agent pipeline.
    """
    print("\n" + "="*60)
    print("Example: Agent Decision Making")
    print("="*60)
    
    from brain import MockLLMClient, Agent
    from core import VisionState, get_state_manager
    
    # Create agent with mock LLM
    llm = MockLLMClient()
    agent = Agent(llm_client=llm)
    
    print("\n--- Scenario 1: Simple movement ---")
    vision = VisionState(obstacle_ahead=False)
    decision = agent.decide("Move forward", vision)
    print(f"  Intent: {decision.intent.value}")
    print(f"  Commands: {len(decision.motor_commands)}")
    print(f"  Reply: {decision.reply_text}")
    
    print("\n--- Scenario 2: Movement blocked by obstacle ---")
    vision = VisionState(obstacle_ahead=True)
    decision = agent.decide("Move forward", vision)
    print(f"  Intent: {decision.intent.value}")
    print(f"  Commands: {len(decision.motor_commands)}")
    print(f"  Reply: {decision.reply_text}")
    
    print("\n--- Scenario 3: Query about obstacles ---")
    vision = VisionState(obstacle_ahead=True, obstacle_distance=0.5)
    decision = agent.decide("Is there anything in front of me?", vision)
    print(f"  Intent: {decision.intent.value}")
    print(f"  Reply: {decision.reply_text}")
    
    print("\n--- Scenario 4: Gesture command ---")
    decision = agent.decide("Jump!", VisionState())
    print(f"  Intent: {decision.intent.value}")
    print(f"  Commands: {[c.to_dict() for c in decision.motor_commands]}")
    print(f"  Reply: {decision.reply_text}")
    
    print("\n--- Scenario 5: Light control ---")
    decision = agent.decide("Turn on red lights", VisionState())
    print(f"  Intent: {decision.intent.value}")
    print(f"  Commands: {[c.to_dict() for c in decision.motor_commands]}")
    print(f"  Reply: {decision.reply_text}")
    
    print("\nAgent example complete!")


def example_full_pipeline():
    """
    Example: Full Pipeline
    
    Demonstrates complete input → brain → output flow.
    """
    print("\n" + "="*60)
    print("Example: Full Pipeline")
    print("="*60)
    
    from brain import MockLLMClient, Agent
    from output import MockMotorController, MockTTS
    from core import VisionState
    
    # Create components
    llm = MockLLMClient()
    agent = Agent(llm_client=llm)
    motor = MockMotorController()
    tts = MockTTS()
    
    motor.connect()
    
    # Simulate a user interaction
    commands = [
        ("Move forward", VisionState(obstacle_ahead=False)),
        ("Move forward", VisionState(obstacle_ahead=True)),
        ("Look around", VisionState()),
        ("Do a jump", VisionState()),
    ]
    
    for text, vision in commands:
        print(f"\n> User says: '{text}'")
        if vision.obstacle_ahead:
            print("  [Vision: Obstacle detected ahead]")
        
        # Agent makes decision
        decision = agent.decide(text, vision)
        
        # Execute motor commands
        if decision.motor_commands:
            print("  Executing motor commands:")
            for cmd in decision.motor_commands:
                motor.execute(cmd)
        
        # Speak response
        if decision.reply_text:
            tts.speak(decision.reply_text)
        
        time.sleep(0.3)
    
    motor.disconnect()
    print("\nFull pipeline example complete!")


def run_all_examples():
    """Run all examples."""
    example_motor_controller()
    example_vision_processor()
    # example_speech_to_text()  # Requires microphone
    example_text_to_speech()
    example_llm_client()
    example_agent()
    example_full_pipeline()


def main():
    """Main entry point for examples."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    example = sys.argv[1].lower()
    
    examples = {
        "motor": example_motor_controller,
        "vision": example_vision_processor,
        "stt": example_speech_to_text,
        "tts": example_text_to_speech,
        "llm": example_llm_client,
        "agent": example_agent,
        "pipeline": example_full_pipeline,
        "all": run_all_examples,
    }
    
    if example in examples:
        examples[example]()
    else:
        print(f"Unknown example: {example}")
        print(f"Available: {', '.join(examples.keys())}")


if __name__ == "__main__":
    main()
