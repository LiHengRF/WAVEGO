"""
Robot State Manager
===================
Manages and tracks the current state of the robot.
Thread-safe state management with observers pattern.
"""

import threading
import time
from typing import Callable, List, Optional
from .command_types import RobotState, VisionState


class RobotStateManager:
    """
    Thread-safe manager for robot state.
    
    Provides a centralized way to track and update robot state,
    with support for state change observers.
    
    Example:
        state_mgr = RobotStateManager()
        state_mgr.add_observer(lambda old, new: print(f"State changed: {new.mode}"))
        state_mgr.update_mode("walking")
    """
    
    def __init__(self):
        """Initialize the state manager."""
        self._lock = threading.RLock()
        self._robot_state = RobotState()
        self._vision_state = VisionState()
        self._observers: List[Callable] = []
    
    @property
    def robot_state(self) -> RobotState:
        """Get current robot state (thread-safe copy)."""
        with self._lock:
            # Return a copy to prevent external modification
            return RobotState(
                mode=self._robot_state.mode,
                last_action=self._robot_state.last_action,
                last_action_time=self._robot_state.last_action_time,
                is_moving=self._robot_state.is_moving,
                current_speed=self._robot_state.current_speed,
                light_color=self._robot_state.light_color,
                steady_mode=self._robot_state.steady_mode
            )
    
    @property
    def vision_state(self) -> VisionState:
        """Get current vision state (thread-safe copy)."""
        with self._lock:
            return VisionState(
                obstacle_ahead=self._vision_state.obstacle_ahead,
                obstacle_distance=self._vision_state.obstacle_distance,
                face_detected=self._vision_state.face_detected,
                face_count=self._vision_state.face_count,
                face_positions=self._vision_state.face_positions.copy(),
                color_target_detected=self._vision_state.color_target_detected,
                color_target_position=self._vision_state.color_target_position,
                color_target_radius=self._vision_state.color_target_radius,
                motion_detected=self._vision_state.motion_detected,
                motion_area=self._vision_state.motion_area,
                timestamp=self._vision_state.timestamp
            )
    
    def add_observer(self, callback: Callable[[RobotState, RobotState], None]):
        """
        Add an observer that will be called when state changes.
        
        Args:
            callback: Function taking (old_state, new_state)
        """
        self._observers.append(callback)
    
    def remove_observer(self, callback: Callable):
        """Remove an observer."""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self, old_state: RobotState, new_state: RobotState):
        """Notify all observers of state change."""
        for observer in self._observers:
            try:
                observer(old_state, new_state)
            except Exception as e:
                print(f"[StateManager] Observer error: {e}")
    
    def update_mode(self, mode: str):
        """Update the operating mode."""
        with self._lock:
            old_state = self.robot_state
            self._robot_state.mode = mode
            self._notify_observers(old_state, self._robot_state)
    
    def update_action(self, action: str):
        """Update the last action."""
        with self._lock:
            old_state = self.robot_state
            self._robot_state.last_action = action
            self._robot_state.last_action_time = time.time()
            self._notify_observers(old_state, self._robot_state)
    
    def update_moving(self, is_moving: bool):
        """Update movement status."""
        with self._lock:
            old_state = self.robot_state
            self._robot_state.is_moving = is_moving
            self._notify_observers(old_state, self._robot_state)
    
    def update_speed(self, speed: int):
        """Update current speed (0-100)."""
        with self._lock:
            old_state = self.robot_state
            self._robot_state.current_speed = max(0, min(100, speed))
            self._notify_observers(old_state, self._robot_state)
    
    def update_light(self, color: str):
        """Update LED color."""
        with self._lock:
            old_state = self.robot_state
            self._robot_state.light_color = color
            self._notify_observers(old_state, self._robot_state)
    
    def update_steady_mode(self, enabled: bool):
        """Update steady/balance mode."""
        with self._lock:
            old_state = self.robot_state
            self._robot_state.steady_mode = enabled
            self._notify_observers(old_state, self._robot_state)
    
    def update_vision(self, vision_state: VisionState):
        """
        Update the entire vision state.
        
        Args:
            vision_state: New vision state from camera processing
        """
        with self._lock:
            self._vision_state = vision_state
    
    def update_obstacle(self, detected: bool, distance: Optional[float] = None):
        """Update obstacle detection state."""
        with self._lock:
            self._vision_state.obstacle_ahead = detected
            self._vision_state.obstacle_distance = distance
            self._vision_state.timestamp = time.time()
    
    def update_faces(self, faces: List[tuple]):
        """Update face detection state."""
        with self._lock:
            self._vision_state.face_detected = len(faces) > 0
            self._vision_state.face_count = len(faces)
            self._vision_state.face_positions = faces
            self._vision_state.timestamp = time.time()
    
    def update_color_target(self, detected: bool, position: tuple = None, radius: float = 0):
        """Update color target detection state."""
        with self._lock:
            self._vision_state.color_target_detected = detected
            self._vision_state.color_target_position = position
            self._vision_state.color_target_radius = radius
            self._vision_state.timestamp = time.time()
    
    def update_motion(self, detected: bool, area: tuple = None):
        """Update motion detection state."""
        with self._lock:
            self._vision_state.motion_detected = detected
            self._vision_state.motion_area = area
            self._vision_state.timestamp = time.time()
    
    def get_full_context(self) -> dict:
        """
        Get combined robot and vision state for LLM context.
        
        Returns:
            Dictionary containing both robot and vision states
        """
        with self._lock:
            return {
                "robot_state": self._robot_state.to_dict(),
                "vision_state": self._vision_state.to_dict()
            }
    
    def reset(self):
        """Reset all state to defaults."""
        with self._lock:
            self._robot_state = RobotState()
            self._vision_state = VisionState()


# Global state manager instance
_state_manager: Optional[RobotStateManager] = None


def get_state_manager() -> RobotStateManager:
    """
    Get the global state manager instance.
    
    Returns:
        The singleton RobotStateManager instance
    """
    global _state_manager
    if _state_manager is None:
        _state_manager = RobotStateManager()
    return _state_manager
