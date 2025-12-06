"""
Command Types and Data Structures
=================================
Defines all command types, states, and data structures used throughout the system.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import time


class Intent(Enum):
    """High-level intent categories."""
    MOVE = "move"
    LOOK = "look"
    STOP = "stop"
    QUERY = "query"
    GESTURE = "gesture"
    LIGHT = "light"
    BUZZER = "buzzer"
    UNKNOWN = "unknown"


class MoveAction(Enum):
    """Movement actions."""
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP_FB = "stop_fb"  # Stop forward/backward
    STOP_LR = "stop_lr"  # Stop left/right
    STOP = "stop"        # Stop all


class LookAction(Enum):
    """Head/look actions."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STOP_UD = "stop_ud"
    STOP_LR = "stop_lr"


class GestureAction(Enum):
    """Gesture/special actions."""
    JUMP = "jump"
    HANDSHAKE = "handshake"
    STEADY = "steady"


class LightColor(Enum):
    """LED light colors."""
    OFF = "off"
    BLUE = "blue"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    CYAN = "cyan"
    MAGENTA = "magenta"
    CYBER = "cyber"


@dataclass
class MotorCommand:
    """
    A single motor command to be sent to ESP32.
    
    Attributes:
        command_type: Type of command (move, ges, funcMode, light, buzzer)
        value: Command value (varies by type)
        duration: How long to execute (seconds), None for instant commands
    """
    command_type: str
    value: int
    duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for ESP32."""
        return {"var": self.command_type, "val": self.value}


@dataclass
class VisionState:
    """
    Current vision/perception state from camera processing.
    
    Attributes:
        obstacle_ahead: Whether an obstacle is detected in front
        obstacle_distance: Estimated distance to obstacle (if available)
        face_detected: Whether a face is detected
        face_count: Number of faces detected
        face_positions: List of (x, y, w, h) for each face
        color_target_detected: Whether target color is detected
        color_target_position: (x, y) center of color target
        color_target_radius: Size/radius of detected color blob
        motion_detected: Whether motion is detected
        motion_area: (x, y, w, h) of motion area
        raw_frame: Optional raw frame data
        timestamp: When this state was captured
    """
    obstacle_ahead: bool = False
    obstacle_distance: Optional[float] = None
    face_detected: bool = False
    face_count: int = 0
    face_positions: List[tuple] = field(default_factory=list)
    color_target_detected: bool = False
    color_target_position: Optional[tuple] = None
    color_target_radius: float = 0
    motion_detected: bool = False
    motion_area: Optional[tuple] = None
    raw_frame: Any = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for LLM context (excludes raw_frame)."""
        return {
            "obstacle_ahead": self.obstacle_ahead,
            "obstacle_distance": self.obstacle_distance,
            "face_detected": self.face_detected,
            "face_count": self.face_count,
            "color_target_detected": self.color_target_detected,
            "color_target_position": self.color_target_position,
            "motion_detected": self.motion_detected
        }


@dataclass
class RobotState:
    """
    Current state of the robot.
    
    Attributes:
        mode: Current operating mode
        last_action: Description of last action taken
        last_action_time: Timestamp of last action
        is_moving: Whether robot is currently moving
        current_speed: Current movement speed (0-100)
        light_color: Current LED color
        steady_mode: Whether steady/balance mode is active
    """
    mode: str = "idle"
    last_action: str = "none"
    last_action_time: float = 0
    is_moving: bool = False
    current_speed: int = 100
    light_color: str = "off"
    steady_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for LLM context."""
        return {
            "mode": self.mode,
            "last_action": self.last_action,
            "is_moving": self.is_moving,
            "current_speed": self.current_speed,
            "light_color": self.light_color,
            "steady_mode": self.steady_mode
        }
    
    def update_action(self, action: str):
        """Update the last action and timestamp."""
        self.last_action = action
        self.last_action_time = time.time()


@dataclass
class AgentDecision:
    """
    Output from the agent's decision-making process.
    
    Attributes:
        intent: Detected intent category
        motor_commands: List of motor commands to execute
        reply_text: Text response for the user
        should_speak: Whether to speak the reply
        success: Whether decision was successful
        error_message: Error description if failed
    """
    intent: Intent
    motor_commands: List[MotorCommand] = field(default_factory=list)
    reply_text: str = ""
    should_speak: bool = True
    success: bool = True
    error_message: Optional[str] = None


# =============================================================================
# Motor Command Mapping
# =============================================================================
# Maps high-level actions to ESP32 command values

MOVE_COMMAND_MAP = {
    MoveAction.FORWARD: 1,
    MoveAction.LEFT: 2,
    MoveAction.STOP_FB: 3,
    MoveAction.RIGHT: 4,
    MoveAction.BACKWARD: 5,
    MoveAction.STOP_LR: 6,
}

LOOK_COMMAND_MAP = {
    LookAction.UP: 1,
    LookAction.DOWN: 2,
    LookAction.STOP_UD: 3,
    LookAction.LEFT: 4,
    LookAction.RIGHT: 5,
    LookAction.STOP_LR: 6,
}

GESTURE_COMMAND_MAP = {
    GestureAction.STEADY: 1,
    GestureAction.HANDSHAKE: 3,
    GestureAction.JUMP: 4,
}

LIGHT_COLOR_MAP = {
    LightColor.OFF: 0,
    LightColor.BLUE: 1,
    LightColor.RED: 2,
    LightColor.GREEN: 3,
    LightColor.YELLOW: 4,
    LightColor.CYAN: 5,
    LightColor.MAGENTA: 6,
    LightColor.CYBER: 7,
}


def create_move_command(action: MoveAction, duration: float = None) -> MotorCommand:
    """Create a movement motor command."""
    return MotorCommand(
        command_type="move",
        value=MOVE_COMMAND_MAP[action],
        duration=duration
    )


def create_look_command(action: LookAction, duration: float = None) -> MotorCommand:
    """Create a head/look motor command."""
    return MotorCommand(
        command_type="ges",
        value=LOOK_COMMAND_MAP[action],
        duration=duration
    )


def create_gesture_command(action: GestureAction) -> MotorCommand:
    """Create a gesture motor command."""
    return MotorCommand(
        command_type="funcMode",
        value=GESTURE_COMMAND_MAP[action],
        duration=None
    )


def create_light_command(color: LightColor) -> MotorCommand:
    """Create an LED light motor command."""
    return MotorCommand(
        command_type="light",
        value=LIGHT_COLOR_MAP[color],
        duration=None
    )


def create_buzzer_command(on: bool) -> MotorCommand:
    """Create a buzzer motor command."""
    return MotorCommand(
        command_type="buzzer",
        value=1 if on else 0,
        duration=None
    )
