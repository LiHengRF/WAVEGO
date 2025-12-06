#!/usr/bin/env python3
"""
Motor Controller Module
=======================
Unified interface for sending commands to the ESP32 motor controller.
Strictly follows the original robot.py interface.

Original robot.py command reference:
- Movement: {"var":"move", "val": 1=forward, 2=left, 3=stopFB, 4=right, 5=backward, 6=stopLR}
- Gesture:  {"var":"ges", "val": 1=lookUp, 2=lookDown, 3=lookStopUD, 4=lookLeft, 5=lookRight, 6=lookStopLR}
- FuncMode: {"var":"funcMode", "val": 1=steadyMode, 3=handShake, 4=jump}
- Light:    {"var":"light", "val": 0-7 (off,blue,red,green,yellow,cyan,magenta,cyber)}
- Buzzer:   {"var":"buzzer", "val": 0=off, 1=on}
"""

import json
import time
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.command_types import MotorCommand


# =============================================================================
# Command Value Constants (from original robot.py)
# =============================================================================

# Movement commands (var: "move")
MOVE_FORWARD = 1
MOVE_LEFT = 2
MOVE_STOP_FB = 3
MOVE_RIGHT = 4
MOVE_BACKWARD = 5
MOVE_STOP_LR = 6

# Gesture/Head commands (var: "ges")
GES_LOOK_UP = 1
GES_LOOK_DOWN = 2
GES_STOP_UD = 3
GES_LOOK_LEFT = 4
GES_LOOK_RIGHT = 5
GES_STOP_LR = 6

# Function mode commands (var: "funcMode")
FUNC_STEADY = 1
FUNC_HANDSHAKE = 3
FUNC_JUMP = 4

# Light colors (var: "light")
LIGHT_OFF = 0
LIGHT_BLUE = 1
LIGHT_RED = 2
LIGHT_GREEN = 3
LIGHT_YELLOW = 4
LIGHT_CYAN = 5
LIGHT_MAGENTA = 6
LIGHT_CYBER = 7  # Original name from robot.py (same as white)

# Color name to value mapping
LIGHT_COLOR_MAP = {
    "off": LIGHT_OFF,
    "blue": LIGHT_BLUE,
    "red": LIGHT_RED,
    "green": LIGHT_GREEN,
    "yellow": LIGHT_YELLOW,
    "cyan": LIGHT_CYAN,
    "magenta": LIGHT_MAGENTA,
    "cyber": LIGHT_CYBER,
    "white": LIGHT_CYBER,  # Alias
}


class MotorController:
    """
    Motor controller for ESP32 communication.
    
    Strictly follows the original robot.py interface.
    Communicates via serial port using JSON commands.
    
    Example:
        controller = MotorController(port="/dev/ttyS0", baudrate=115200)
        controller.connect()
        
        # Movement (matches robot.forward(), robot.backward(), etc.)
        controller.forward(speed=100)
        controller.left(speed=100)
        controller.stop_fb()
        controller.stop_lr()
        
        # Head/Gesture (matches robot.lookUp(), robot.lookDown(), etc.)
        controller.look_up()
        controller.look_down()
        controller.look_stop_ud()
        
        # Function modes
        controller.jump()
        controller.handshake()
        controller.steady_mode()
        
        # Light control (two lights supported via light_id)
        controller.light_ctrl("red", light_id=0)
        controller.light_ctrl("blue", light_id=1)
        
        # Buzzer
        controller.buzzer_ctrl(on=True, buzzer_id=0)
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
            port: Serial port path (default: /dev/ttyS0)
            baudrate: Serial baudrate (default: 115200)
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
        
        # For timed commands
        self._stop_event = threading.Event()
        self._command_thread: Optional[threading.Thread] = None
        
        # Current speed (matches original robot.py speedMove)
        self._speed = 100
    
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
            print(f"[Motor] Not connected, would send: {json.dumps(data)}")
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
        Execute a MotorCommand object.
        
        Args:
            command: MotorCommand to execute
            
        Returns:
            True if executed successfully
        """
        success = self.send_raw(command.to_dict())
        
        if success and command.duration and command.duration > 0:
            self._schedule_stop(command, command.duration)
        
        return success
    
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
            self.stop_fb()
            self.stop_lr()
        elif command.command_type == "ges":
            self.look_stop_ud()
            self.look_stop_lr()
    
    # ==========================================================================
    # Movement Commands - Match original robot.py exactly
    # ==========================================================================
    
    def forward(self, speed: int = 100, duration: Optional[float] = None):
        """
        Move forward.
        
        Args:
            speed: Movement speed 0-100 (matches original interface)
            duration: Optional duration in seconds
        """
        self._speed = speed
        self.send_raw({"var": "move", "val": MOVE_FORWARD})
        print("robot-forward")
        
        if duration:
            time.sleep(duration)
            self.stop_fb()
    
    def backward(self, speed: int = 100, duration: Optional[float] = None):
        """
        Move backward.
        
        Args:
            speed: Movement speed 0-100
            duration: Optional duration in seconds
        """
        self._speed = speed
        self.send_raw({"var": "move", "val": MOVE_BACKWARD})
        print("robot-backward")
        
        if duration:
            time.sleep(duration)
            self.stop_fb()
    
    def left(self, speed: int = 100, duration: Optional[float] = None):
        """
        Turn left.
        
        Args:
            speed: Movement speed 0-100
            duration: Optional duration in seconds
        """
        self._speed = speed
        self.send_raw({"var": "move", "val": MOVE_LEFT})
        print("robot-left")
        
        if duration:
            time.sleep(duration)
            self.stop_lr()
    
    def right(self, speed: int = 100, duration: Optional[float] = None):
        """
        Turn right.
        
        Args:
            speed: Movement speed 0-100
            duration: Optional duration in seconds
        """
        self._speed = speed
        self.send_raw({"var": "move", "val": MOVE_RIGHT})
        print("robot-right")
        
        if duration:
            time.sleep(duration)
            self.stop_lr()
    
    def stop_lr(self):
        """Stop left/right movement (matches robot.stopLR())."""
        self.send_raw({"var": "move", "val": MOVE_STOP_LR})
        print("robot-stop")
    
    def stop_fb(self):
        """Stop forward/backward movement (matches robot.stopFB())."""
        self.send_raw({"var": "move", "val": MOVE_STOP_FB})
        print("robot-stop")
    
    def stop(self):
        """Stop all movement."""
        self._stop_event.set()
        self.stop_fb()
        self.stop_lr()
        self._stop_event.clear()
    
    # ==========================================================================
    # Head/Gesture Commands - Match original robot.py exactly
    # ==========================================================================
    
    def look_up(self, duration: Optional[float] = None):
        """Look up (matches robot.lookUp())."""
        self.send_raw({"var": "ges", "val": GES_LOOK_UP})
        print("robot-lookUp")
        
        if duration:
            time.sleep(duration)
            self.look_stop_ud()
    
    def look_down(self, duration: Optional[float] = None):
        """Look down (matches robot.lookDown())."""
        self.send_raw({"var": "ges", "val": GES_LOOK_DOWN})
        print("robot-lookDown")
        
        if duration:
            time.sleep(duration)
            self.look_stop_ud()
    
    def look_stop_ud(self):
        """Stop up/down head movement (matches robot.lookStopUD())."""
        self.send_raw({"var": "ges", "val": GES_STOP_UD})
        print("robot-lookStopUD")
    
    def look_left(self, duration: Optional[float] = None):
        """Look left (matches robot.lookLeft())."""
        self.send_raw({"var": "ges", "val": GES_LOOK_LEFT})
        print("robot-lookLeft")
        
        if duration:
            time.sleep(duration)
            self.look_stop_lr()
    
    def look_right(self, duration: Optional[float] = None):
        """Look right (matches robot.lookRight())."""
        self.send_raw({"var": "ges", "val": GES_LOOK_RIGHT})
        print("robot-lookRight")
        
        if duration:
            time.sleep(duration)
            self.look_stop_lr()
    
    def look_stop_lr(self):
        """Stop left/right head movement (matches robot.lookStopLR())."""
        self.send_raw({"var": "ges", "val": GES_STOP_LR})
        print("robot-lookStopLR")
    
    def stop_head(self):
        """Stop all head movement."""
        self.look_stop_ud()
        self.look_stop_lr()
    
    # ==========================================================================
    # Function Mode Commands - Match original robot.py exactly
    # ==========================================================================
    
    def steady_mode(self):
        """Enable steady/balance mode (matches robot.steadyMode())."""
        self.send_raw({"var": "funcMode", "val": FUNC_STEADY})
        print("robot-steady")
    
    def jump(self):
        """Perform jump gesture (matches robot.jump())."""
        self.send_raw({"var": "funcMode", "val": FUNC_JUMP})
        print("robot-jump")
    
    def handshake(self):
        """Perform handshake gesture (matches robot.handShake())."""
        self.send_raw({"var": "funcMode", "val": FUNC_HANDSHAKE})
        print("robot-handshake")
    
    # ==========================================================================
    # Light Control - Match original robot.py exactly
    # Original: def lightCtrl(colorName, cmdInput)
    # Note: WaveGo has TWO lights, cmdInput selects which light (0 or 1)
    # ==========================================================================
    
    def light_ctrl(self, color_name: str, light_id: int = 0):
        """
        Control LED light color.
        
        Matches original robot.lightCtrl(colorName, cmdInput).
        WaveGo has two independent lights that can be controlled separately.
        
        Args:
            color_name: Color name (off, blue, red, green, yellow, cyan, magenta, cyber)
            light_id: Which light to control (0 or 1). Default 0 for backward compatibility.
        
        Example:
            # Set light 0 to red
            controller.light_ctrl("red", 0)
            
            # Set light 1 to blue
            controller.light_ctrl("blue", 1)
            
            # Turn off both lights
            controller.light_ctrl("off", 0)
            controller.light_ctrl("off", 1)
        """
        color_num = LIGHT_COLOR_MAP.get(color_name.lower(), LIGHT_OFF)
        
        # Use different var names for different lights
        # light_id=0 -> "light" (front/main light)
        # light_id=1 -> "lightB" (back/secondary light)
        if light_id == 0:
            self.send_raw({"var": "light", "val": color_num})
        else:
            self.send_raw({"var": "lightB", "val": color_num})
    
    def set_light(self, color: str, light_id: int = 0):
        """
        Alias for light_ctrl for web interface compatibility.
        
        Args:
            color: Color name
            light_id: Which light (0 or 1)
        """
        self.light_ctrl(color, light_id)
    
    def lights_off(self):
        """Turn off all lights."""
        self.light_ctrl("off", 0)
        self.light_ctrl("off", 1)
    
    # ==========================================================================
    # Buzzer Control - Match original robot.py exactly
    # Original: def buzzerCtrl(buzzerCtrl, cmdInput)
    # ==========================================================================
    
    def buzzer_ctrl(self, on: bool, buzzer_id: int = 0):
        """
        Control buzzer.
        
        Matches original robot.buzzerCtrl(buzzerCtrl, cmdInput).
        
        Args:
            on: True to turn on, False to turn off
            buzzer_id: Which buzzer (0 by default, reserved for future use)
        """
        val = 1 if on else 0
        self.send_raw({"var": "buzzer", "val": val})
    
    def buzzer(self, on: bool):
        """Alias for buzzer_ctrl for backward compatibility."""
        self.buzzer_ctrl(on, 0)
    
    # ==========================================================================
    # Speed Control
    # ==========================================================================
    
    def set_speed(self, speed: int):
        """
        Set movement speed.
        
        Args:
            speed: Speed value 0-100
        """
        self._speed = max(0, min(100, speed))
    
    def get_speed(self) -> int:
        """Get current speed setting."""
        return self._speed


class MockMotorController(MotorController):
    """
    Mock motor controller for testing without hardware.
    
    Prints commands instead of sending them via serial.
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
        self._speed = 100
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
    
    hardware_config = config.get("hardware", {})
    serial_config = hardware_config.get("serial", {})
    
    controller = MotorController(
        port=serial_config.get("port", "/dev/ttyS0"),
        baudrate=serial_config.get("baudrate", 115200),
        timeout=serial_config.get("timeout", 1.0),
        config=config
    )
    
    return controller
