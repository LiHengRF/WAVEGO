#!/usr/bin/env python3
"""
WaveGo Agent - Web Dashboard Server
Real-time monitoring interface with video streaming, status display, and command input.
"""

import os
import sys
import json
import time
import queue
import base64
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConversationEntry:
    """Single conversation entry"""
    timestamp: str
    role: str  # 'user' or 'assistant'
    text: str
    intent: str = ""
    success: bool = True


@dataclass 
class SystemLog:
    """System log entry"""
    timestamp: str
    level: str  # 'info', 'warning', 'error', 'debug'
    module: str
    message: str


@dataclass
class DashboardState:
    """Complete dashboard state for frontend"""
    # Robot state
    mode: str = "idle"
    is_moving: bool = False
    current_speed: int = 80
    light_color: str = "off"
    steady_mode: bool = False
    last_action: str = "none"
    last_action_time: str = ""
    
    # Vision state
    obstacle_ahead: bool = False
    obstacle_distance: float = 0.0
    face_detected: bool = False
    face_count: int = 0
    color_target_detected: bool = False
    motion_detected: bool = False
    
    # System state
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    temperature: float = 0.0
    uptime: str = "00:00:00"
    
    # Connection state
    serial_connected: bool = False
    camera_active: bool = False
    llm_connected: bool = False
    stt_active: bool = False
    
    # Conversation history (last 50 entries)
    conversation: List[Dict] = field(default_factory=list)
    
    # System logs (last 100 entries)
    logs: List[Dict] = field(default_factory=list)


# =============================================================================
# Dashboard Manager
# =============================================================================

class DashboardManager:
    """Manages dashboard state and provides updates to web clients"""
    
    def __init__(self):
        self.state = DashboardState()
        self.lock = threading.RLock()
        self.frame_queue = queue.Queue(maxsize=2)
        self.update_callbacks = []
        self.start_time = time.time()
        
        # References to agent components (set externally)
        self.agent = None
        self.motor_controller = None
        self.vision_processor = None
        self.stt = None
        self.tts = None
        self.state_manager = None
        
    def set_components(self, agent=None, motor=None, vision=None, 
                       stt=None, tts=None, state_manager=None):
        """Set references to agent components"""
        self.agent = agent
        self.motor_controller = motor
        self.vision_processor = vision
        self.stt = stt
        self.tts = tts
        self.state_manager = state_manager
        
    def update_robot_state(self, **kwargs):
        """Update robot state fields"""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
            self.state.last_action_time = datetime.now().strftime("%H:%M:%S")
                    
    def update_vision_state(self, vision_state: Dict):
        """Update vision state from VisionState dict"""
        with self.lock:
            self.state.obstacle_ahead = vision_state.get('obstacle_ahead', False)
            self.state.obstacle_distance = vision_state.get('obstacle_distance', 0.0)
            self.state.face_detected = vision_state.get('face_detected', False)
            self.state.face_count = vision_state.get('face_count', 0)
            self.state.color_target_detected = vision_state.get('color_target_detected', False)
            self.state.motion_detected = vision_state.get('motion_detected', False)
            
    def update_system_state(self):
        """Update system metrics"""
        try:
            import psutil
            with self.lock:
                self.state.cpu_usage = psutil.cpu_percent()
                self.state.memory_usage = psutil.virtual_memory().percent
                
                # Try to get temperature (Raspberry Pi)
                try:
                    temp_file = '/sys/class/thermal/thermal_zone0/temp'
                    if os.path.exists(temp_file):
                        with open(temp_file) as f:
                            self.state.temperature = int(f.read()) / 1000.0
                except:
                    pass
                    
                # Calculate uptime
                elapsed = int(time.time() - self.start_time)
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)
                self.state.uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except ImportError:
            pass
            
    def update_connection_state(self):
        """Update connection status of components"""
        with self.lock:
            if self.motor_controller:
                self.state.serial_connected = getattr(
                    self.motor_controller, 'is_connected', lambda: False
                )() if callable(getattr(self.motor_controller, 'is_connected', None)) else True
            if self.vision_processor:
                self.state.camera_active = getattr(
                    self.vision_processor, 'is_running', False
                )
            if self.agent and hasattr(self.agent, 'llm_client'):
                self.state.llm_connected = True
            if self.stt:
                self.state.stt_active = getattr(self.stt, 'is_listening', False)
                
    def add_conversation(self, role: str, text: str, intent: str = "", success: bool = True):
        """Add conversation entry"""
        entry = ConversationEntry(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            role=role,
            text=text,
            intent=intent,
            success=success
        )
        with self.lock:
            self.state.conversation.append(asdict(entry))
            # Keep last 50 entries
            if len(self.state.conversation) > 50:
                self.state.conversation = self.state.conversation[-50:]
                
    def add_log(self, level: str, module: str, message: str):
        """Add system log entry"""
        entry = SystemLog(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            level=level,
            module=module,
            message=message
        )
        with self.lock:
            self.state.logs.append(asdict(entry))
            # Keep last 100 entries
            if len(self.state.logs) > 100:
                self.state.logs = self.state.logs[-100:]
                
    def push_frame(self, frame):
        """Push video frame to queue (drops old frames if full)"""
        try:
            # Drop old frame if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass
            
    def get_frame(self, timeout: float = 0.5):
        """Get latest video frame"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_state_dict(self) -> Dict:
        """Get complete state as dictionary"""
        with self.lock:
            self.update_system_state()
            self.update_connection_state()
            return asdict(self.state)


# =============================================================================
# Flask Application
# =============================================================================

# Global dashboard manager
dashboard = DashboardManager()

# Create Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
CORS(app)

# Logging handler that sends to dashboard
class DashboardLogHandler(logging.Handler):
    def emit(self, record):
        try:
            dashboard.add_log(
                level=record.levelname.lower(),
                module=record.name,
                message=record.getMessage()
            )
        except:
            pass


@app.route('/')
def index():
    """Serve main dashboard page"""
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    """Get current dashboard state"""
    return jsonify(dashboard.get_state_dict())


@app.route('/api/command', methods=['POST'])
def send_command():
    """Process text command"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'success': False, 'error': 'No command provided'})
    
    # Add user message to conversation
    dashboard.add_conversation('user', text)
    dashboard.add_log('info', 'web', f'Command received: {text}')
    
    # Process command if agent is available
    if dashboard.agent:
        try:
            # Get vision state
            vision_state = None
            if dashboard.vision_processor:
                vision_state = dashboard.vision_processor.get_vision_state()
            
            # Get decision from agent
            decision = dashboard.agent.decide(text, vision_state)
            
            # Execute motor commands
            if dashboard.motor_controller and decision.motor_commands:
                for cmd in decision.motor_commands:
                    dashboard.motor_controller.execute(cmd)
                    dashboard.add_log('info', 'motor', f'Executed: {cmd.command_type}')
            
            # Update state
            dashboard.update_robot_state(
                last_action=decision.intent.value if decision.intent else 'unknown',
                mode='active'
            )
            
            # Add response to conversation
            dashboard.add_conversation(
                'assistant', 
                decision.reply_text,
                intent=decision.intent.value if decision.intent else '',
                success=decision.success
            )
            
            # Speak response if TTS available
            if dashboard.tts and decision.reply_text:
                dashboard.tts.speak_async(decision.reply_text)
            
            return jsonify({
                'success': True,
                'reply': decision.reply_text,
                'intent': decision.intent.value if decision.intent else None
            })
            
        except Exception as e:
            error_msg = f'Error processing command: {str(e)}'
            dashboard.add_log('error', 'agent', error_msg)
            dashboard.add_conversation('assistant', error_msg, success=False)
            return jsonify({'success': False, 'error': error_msg})
    else:
        # No agent - just echo for testing
        reply = f"[Mock] Received: {text}"
        dashboard.add_conversation('assistant', reply, intent='mock')
        return jsonify({'success': True, 'reply': reply, 'intent': 'mock'})


@app.route('/api/motor/<action>', methods=['POST'])
def motor_action(action: str):
    """Direct motor control"""
    valid_actions = ['forward', 'backward', 'left', 'right', 'stop',
                     'look_up', 'look_down', 'look_left', 'look_right',
                     'jump', 'handshake', 'steady_mode']
    
    if action not in valid_actions:
        return jsonify({'success': False, 'error': f'Invalid action: {action}'})
    
    dashboard.add_log('info', 'web', f'Motor action: {action}')
    
    if dashboard.motor_controller:
        try:
            method = getattr(dashboard.motor_controller, action, None)
            if method:
                method()
                dashboard.update_robot_state(last_action=action)
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'error': 'Method not found'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        # Mock mode
        dashboard.update_robot_state(last_action=action)
        return jsonify({'success': True, 'mock': True})


@app.route('/api/light/<color>', methods=['POST'])
def set_light(color: str):
    """Set LED light color"""
    valid_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white', 'off']
    
    if color not in valid_colors:
        return jsonify({'success': False, 'error': f'Invalid color: {color}'})
    
    dashboard.add_log('info', 'web', f'Light color: {color}')
    dashboard.update_robot_state(light_color=color)
    
    if dashboard.motor_controller:
        try:
            dashboard.motor_controller.set_light(color)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({'success': True, 'mock': True})


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame = dashboard.get_frame(timeout=1.0)
            if frame is not None:
                # Encode frame as JPEG
                if CV2_AVAILABLE:
                    _, buffer = cv2.imencode('.jpg', frame, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # No frame available, send placeholder
                time.sleep(0.1)
                
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# =============================================================================
# Integration with WaveGo Agent
# =============================================================================

def create_dashboard_server(agent=None, motor=None, vision=None, 
                           stt=None, tts=None, state_manager=None,
                           host='0.0.0.0', port=5000):
    """
    Create and configure dashboard server with agent components.
    
    Returns:
        (app, dashboard, run_func): Flask app, dashboard manager, and run function
    """
    dashboard.set_components(agent, motor, vision, stt, tts, state_manager)
    
    # Set up vision frame callback
    if vision:
        def on_frame(frame):
            dashboard.push_frame(frame)
        # Assuming vision processor has a frame callback mechanism
        if hasattr(vision, 'set_frame_callback'):
            vision.set_frame_callback(on_frame)
    
    # Set up logging integration
    root_logger = logging.getLogger()
    root_logger.addHandler(DashboardLogHandler())
    
    def run():
        app.run(host=host, port=port, threaded=True, use_reloader=False)
        
    return app, dashboard, run


def run_standalone(host='0.0.0.0', port=5000, mock=True):
    """Run dashboard in standalone mode for testing"""
    print(f"Starting WaveGo Dashboard at http://{host}:{port}")
    print("Running in mock mode - no hardware connected")
    
    # Add some mock data
    dashboard.add_log('info', 'system', 'Dashboard started in mock mode')
    dashboard.add_conversation('user', 'Hello robot!')
    dashboard.add_conversation('assistant', 'Hello! I am WaveGo robot. How can I help you?', 
                              intent='greeting')
    
    # Mock state updates
    def mock_updates():
        import random
        while True:
            dashboard.update_robot_state(
                mode='idle' if random.random() > 0.3 else 'active',
                obstacle_ahead=random.random() > 0.7,
                face_detected=random.random() > 0.8,
                face_count=random.randint(0, 2) if random.random() > 0.5 else 0
            )
            dashboard.update_system_state()
            time.sleep(2)
    
    mock_thread = threading.Thread(target=mock_updates, daemon=True)
    mock_thread.start()
    
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='WaveGo Dashboard Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    args = parser.parse_args()
    
    run_standalone(host=args.host, port=args.port)
