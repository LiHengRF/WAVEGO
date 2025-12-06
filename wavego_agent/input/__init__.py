"""
Input Module
============
Handles all input sources for the WaveGo Agent:
- Speech-to-Text (voice commands)
- Vision Processing (camera/OpenCV)
"""

from .speech_to_text import SpeechToText, create_stt
from .vision_processor import VisionProcessor, create_vision_processor

__all__ = [
    "SpeechToText",
    "create_stt",
    "VisionProcessor", 
    "create_vision_processor",
]
