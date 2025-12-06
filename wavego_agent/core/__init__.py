"""
Core Module
===========
Core data structures and state management for the WaveGo Agent.
"""

from .command_types import (
    Intent,
    MoveAction,
    LookAction,
    GestureAction,
    LightColor,
    MotorCommand,
    VisionState,
    RobotState,
    AgentDecision,
    create_move_command,
    create_look_command,
    create_gesture_command,
    create_light_command,
    create_buzzer_command,
)

from .robot_state import (
    RobotStateManager,
    get_state_manager,
)

__all__ = [
    # Enums
    "Intent",
    "MoveAction",
    "LookAction",
    "GestureAction",
    "LightColor",
    # Data classes
    "MotorCommand",
    "VisionState",
    "RobotState",
    "AgentDecision",
    # Command factories
    "create_move_command",
    "create_look_command",
    "create_gesture_command",
    "create_light_command",
    "create_buzzer_command",
    # State management
    "RobotStateManager",
    "get_state_manager",
]
