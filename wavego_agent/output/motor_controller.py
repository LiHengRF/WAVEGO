"""
Motor Controller Module
=======================
Unified interface for sending commands to the ESP32 motor controller.
Wraps the existing robot.py functionality with the new command types.
"""

import json
import time
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.command_types import MotorCommand, MoveAction, MOVE_COMMAND_MAP


class MotorController:
    """
    Motor controller for ESP32 communication.
    
    Provides a unified interface for sending motor commands via serial
    to the ESP32 lower-level controller.
    
    Example:
        controller = MotorController(port="/dev/ttyS0", baudrate=115200)
        controller.connect()
        
        # Execute a command
        cmd = MotorCommand(command_type="move", value=1, duration=1.0)
        controller.execute(cmd)
        
        # Or use convenience methods
        controller.forward(duration=1.0)
        controller.stop()
    """
    
    def __init__(
        self,
        port: str = "/dev/ttyS0",
        baudrate: int = 115200,
        timeout: float = 1.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the motor controller.
        
        Args:
            port: Serial port path
            baudrate: Serial baudrate
            timeout: Serial timeout in seconds
            config: Additional configuration
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.config = config or {}
        
        self._serial = None
        self._lock = threading.Lock()
        self._connected = False
        
        # Command execution thread for timed commands
        self._command_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def connect(self) -> bool:
        """
        Connect to the ESP32 via serial.
        
        Returns:
            True if connected successfully
        """
        if self._connected:
            return True
        
        try:
            import serial
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self._connected = True
            print(f"[Motor] Connected to {self.port} at {self.baudrate} baud")
            return True
        except ImportError:
            print("[Motor] pyserial not installed. Install with: pip install pyserial")
            return False
        except Exception as e:
            print(f"[Motor] Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial port."""
        self._stop_event.set()
        
        if self._command_thread:
            self._command_thread.join(timeout=2.0)
        
        if self._serial:
            self._serial.close()
            self._serial = None
        
        self._connected = False
        print("[Motor] Disconnected")
    
    def is_connected(self) -> bool:
        """Check if connected to ESP32."""
        return self._connected and self._serial is not None
    
    def send_raw(self, data: Dict[str, Any]) -> bool:
        """
        Send raw JSON command to ESP32.
        
        Args:
            data: Dictionary to serialize and send
            
        Returns:
            True if sent successfully
        """
        if not self.is_connected():
            print("[Motor] Not connected")
            return False
        
        try:
            with self._lock:
                cmd_str = json.dumps(data)
                self._serial.write(cmd_str.encode())
                print(f"[Motor] Sent: {cmd_str}")
                return True
        except Exception as e:
            print(f"[Motor] Send failed: {e}")
            return False
    
    def execute(self, command: MotorCommand) -> bool:
        """
        Execute a motor command.
        
        Args:
            command: MotorCommand to execute
            
        Returns:
            True if executed successfully
        """
        # Send the command
        success = self.send_raw(command.to_dict())
        
        if not success:
            return False
        
        # If command has duration, schedule stop
        if command.duration and command.duration > 0:
            self._schedule_stop(command, command.duration)
        
        return True
    
    def execute_sequence(self, commands: List[MotorCommand], delay: float = 0.1) -> bool:
        """
        Execute a sequence of commands.
        
        Args:
            commands: List of commands to execute
            delay: Delay between commands in seconds
            
        Returns:
            True if all commands executed successfully
        """
        for i, cmd in enumerate(commands):
            success = self.execute(cmd)
            if not success:
                return False
            
            # Wait for duration if specified
            if cmd.duration:
                time.sleep(cmd.duration)
            elif i < len(commands) - 1:
                time.sleep(delay)
        
        return True
    
    def _schedule_stop(self, command: MotorCommand, duration: float):
        """Schedule a stop command after duration."""
        def delayed_stop():
            time.sleep(duration)
            if not self._stop_event.is_set():
                self._send_stop_for_command(command)
        
        self._command_thread = threading.Thread(target=delayed_stop, daemon=True)
        self._command_thread.start()
    
    def _send_stop_for_command(self, command: MotorCommand):
        """Send appropriate stop command for a movement command."""
        if command.command_type == "move":
            # Stop forward/backward
            self.send_raw({"var": "move", "val": 3})
            # Stop left/right
            self.send_raw({"var": "move", "val": 6})
        elif command.command_type == "ges":
            # Stop head movement
            self.send_raw({"var": "ges", "val": 3})
            self.send_raw({"var": "ges", "val": 6})
    
    # ==========================================================================
    # Convenience Methods - Match original robot.py interface
    # ==========================================================================
    
    def forward(self, duration: Optional[float] = None):
        """Move forward."""
        cmd = MotorCommand("move", 1, duration)
        self.execute(cmd)
    
    def backward(self, duration: Optional[float] = None):
        """Move backward."""
        cmd = MotorCommand("move", 5, duration)
        self.execute(cmd)
    
    def left(self, duration: Optional[float] = None):
        """Turn left."""
        cmd = MotorCommand("move", 2, duration)
        self.execute(cmd)
    
    def right(self, duration: Optional[float] = None):
        """Turn right."""
        cmd = MotorCommand("move", 4, duration)
        self.execute(cmd)
    
    def stop(self):
        """Stop all movement."""
        self._stop_event.set()
        self.send_raw({"var": "move", "val": 3})  # Stop FB
        self.send_raw({"var": "move", "val": 6})  # Stop LR
        self._stop_event.clear()
    
    def look_up(self, duration: Optional[float] = None):
        """Look up."""
        cmd = MotorCommand("ges", 1, duration)
        self.execute(cmd)
    
    def look_down(self, duration: Optional[float] = None):
        """Look down."""
        cmd = MotorCommand("ges", 2, duration)
        self.execute(cmd)
    
    def look_left(self, duration: Optional[float] = None):
        """Look left."""
        cmd = MotorCommand("ges", 4, duration)
        self.execute(cmd)
    
    def look_right(self, duration: Optional[float] = None):
        """Look right."""
        cmd = MotorCommand("ges", 5, duration)
        self.execute(cmd)
    
    def stop_head(self):
        """Stop head movement."""
        self.send_raw({"var": "ges", "val": 3})  # Stop UD
        self.send_raw({"var": "ges", "val": 6})  # Stop LR
    
    def jump(self):
        """Perform jump gesture."""
        self.send_raw({"var": "funcMode", "val": 4})
    
    def handshake(self):
        """Perform handshake gesture."""
        self.send_raw({"var": "funcMode", "val": 3})
    
    def steady_mode(self):
        """Enable steady/balance mode."""
        self.send_raw({"var": "funcMode", "val": 1})
    
    def set_light(self, color: str):
        """
        Set LED light color.
        
        Args:
            color: Color name (off, blue, red, green, yellow, cyan, magenta, cyber)
        """
        color_map = {
            "off": 0,
            "blue": 1,
            "red": 2,
            "green": 3,
            "yellow": 4,
            "cyan": 5,
            "magenta": 6,
            "cyber": 7
        }
        val = color_map.get(color.lower(), 1)
        self.send_raw({"var": "light", "val": val})
    
    def buzzer(self, on: bool):
        """
        Control buzzer.
        
        Args:
            on: True to turn on, False to turn off
        """
        self.send_raw({"var": "buzzer", "val": 1 if on else 0})


class MockMotorController(MotorController):
    """
    Mock motor controller for testing without hardware.
    
    Prints commands instead of sending them.
    """
    
    def __init__(self, **kwargs):
        """Initialize mock controller."""
        self.port = "mock"
        self.baudrate = 0
        self._connected = True
        self._serial = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._command_thread = None
        self.config = {}
    
    def connect(self) -> bool:
        print("[MockMotor] Connected (mock)")
        return True
    
    def disconnect(self):
        print("[MockMotor] Disconnected (mock)")
    
    def is_connected(self) -> bool:
        return True
    
    def send_raw(self, data: Dict[str, Any]) -> bool:
        print(f"[MockMotor] Would send: {json.dumps(data)}")
        return True


# Factory function
def create_motor_controller(config: Dict[str, Any], use_mock: bool = False) -> MotorController:
    """
    Create a motor controller from configuration.
    
    Args:
        config: Configuration dictionary
        use_mock: If True, create a mock controller
        
    Returns:
        Configured MotorController instance
    """
    if use_mock:
        return MockMotorController()
    
    serial_config = config.get("hardware", {}).get("serial", {})
    
    controller = MotorController(
        port=serial_config.get("port", "/dev/ttyS0"),
        baudrate=serial_config.get("baudrate", 115200),
        timeout=serial_config.get("timeout", 1.0),
        config=config
    )
    
    return controller
