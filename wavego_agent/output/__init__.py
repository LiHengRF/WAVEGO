"""
Output Module
=============
Handles all outputs from the WaveGo Agent:
- Motor Control (ESP32 communication)
- Text-to-Speech (voice responses)
"""

from .motor_controller import MotorController, MockMotorController, create_motor_controller
from .text_to_speech import TextToSpeech, MockTTS, create_tts

__all__ = [
    "MotorController",
    "MockMotorController",
    "create_motor_controller",
    "TextToSpeech",
    "MockTTS",
    "create_tts",
]
