"""
Vision Processor Module
=======================
Handles camera input and computer vision processing including:
- Obstacle detection
- Face detection  
- Color-based object detection
- Motion detection

Designed to integrate with existing camera_opencv.py structure.
Supports both OpenCV VideoCapture and Picamera2 for Raspberry Pi.
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Dict, Any, List, Tuple
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.command_types import VisionState


class VisionProcessor:
    """
    Main vision processing class.
    
    Processes camera frames to extract structured vision information
    like obstacle detection, face detection, and color tracking.
    
    Example:
        processor = VisionProcessor(camera_id=0)
        processor.start()
        
        # Get current vision state
        state = processor.get_vision_state()
        print(f"Obstacle ahead: {state.obstacle_ahead}")
        
        # Or use callback
        processor.set_callback(lambda state: print(state.to_dict()))
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        config: Optional[Dict[str, Any]] = None,
        use_picamera2: bool = False
    ):
        """
        Initialize the vision processor.
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            config: Vision configuration dictionary
            use_picamera2: Use Picamera2 instead of OpenCV (for Raspberry Pi)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.config = config or {}
        self.use_picamera2 = use_picamera2
        
        # Camera
        self._camera: Optional[cv2.VideoCapture] = None
        self._picam2 = None  # Picamera2 instance
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        
        # Processing state
        self._is_running = False
        self._process_thread: Optional[threading.Thread] = None
        self._vision_state = VisionState()
        self._state_lock = threading.Lock()
        
        # Callback for state updates
        self._callback: Optional[Callable[[VisionState], None]] = None
        
        # Face detection
        self._face_cascade = None
        self._init_face_cascade()
        
        # Motion detection background
        self._motion_avg = None
        
        # Color detection settings
        self._color_lower = np.array([24, 100, 100])
        self._color_upper = np.array([44, 255, 255])
    
    def _init_face_cascade(self):
        """Initialize the face detection cascade classifier."""
        # Try multiple paths for the cascade file
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'),
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                self._face_cascade = cv2.CascadeClassifier(path)
                print(f"[Vision] Loaded face cascade from: {path}")
                break
        
        if self._face_cascade is None:
            print("[Vision] Warning: Could not load face cascade classifier")
    
    def set_color_target(self, h_center: int, h_range: int = 15,
                         s_min: int = 100, s_max: int = 255,
                         v_min: int = 100, v_max: int = 255):
        """
        Set the target color for color detection (HSV).
        
        Args:
            h_center: Center hue value (0-180)
            h_range: Range around center hue
            s_min, s_max: Saturation range
            v_min, v_max: Value range
        """
        h_low = max(0, h_center - h_range)
        h_high = min(180, h_center + h_range)
        
        self._color_lower = np.array([h_low, s_min, v_min])
        self._color_upper = np.array([h_high, s_max, v_max])
        print(f"[Vision] Color target set: H={h_center}±{h_range}, "
              f"S={s_min}-{s_max}, V={v_min}-{v_max}")
    
    def set_callback(self, callback: Callable[[VisionState], None]):
        """Set callback for vision state updates."""
        self._callback = callback
    
    def start(self) -> bool:
        """
        Start the vision processor.
        
        Returns:
            True if started successfully
        """
        if self._is_running:
            print("[Vision] Already running")
            return True
        
        # Initialize camera
        if self.use_picamera2:
            # Use Picamera2 for Raspberry Pi
            try:
                from picamera2 import Picamera2
                self._picam2 = Picamera2()
                config = self._picam2.create_preview_configuration(
                    main={"size": (self.width, self.height)}
                )
                self._picam2.configure(config)
                self._picam2.start()
                print(f"[Vision] Picamera2 started ({self.width}x{self.height})")
            except ImportError:
                print("[Vision] Picamera2 not available, falling back to OpenCV")
                self.use_picamera2 = False
            except Exception as e:
                print(f"[Vision] Picamera2 failed: {e}, falling back to OpenCV")
                self.use_picamera2 = False
        
        if not self.use_picamera2:
            # Use OpenCV VideoCapture
            self._camera = cv2.VideoCapture(self.camera_id)
            if not self._camera.isOpened():
                print(f"[Vision] Failed to open camera {self.camera_id}")
                return False
            
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            print(f"[Vision] OpenCV camera started ({self.width}x{self.height})")
        
        # Start processing thread
        self._is_running = True
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        print("[Vision] Started")
        return True
    
    def stop(self):
        """Stop the vision processor."""
        self._is_running = False
        
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None
        
        if self._camera:
            self._camera.release()
            self._camera = None
        
        if self._picam2:
            try:
                self._picam2.stop()
            except:
                pass
            self._picam2 = None
        
        print("[Vision] Stopped")
    
    def get_vision_state(self) -> VisionState:
        """Get the current vision state (thread-safe)."""
        with self._state_lock:
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
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current camera frame (thread-safe)."""
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
            return None
    
    def get_jpeg_frame(self) -> Optional[bytes]:
        """Get the current frame as JPEG bytes."""
        frame = self.get_frame()
        if frame is not None:
            _, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        return None
    
    def _process_loop(self):
        """Main processing loop (runs in background thread)."""
        while self._is_running:
            frame = None
            
            # Capture frame from appropriate source
            if self.use_picamera2 and self._picam2:
                try:
                    # Picamera2 returns XBGR or RGB, convert to BGR for OpenCV
                    frame = self._picam2.capture_array()
                    if frame is not None:
                        # Convert from RGBA/BGRA to BGR if needed
                        if frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                except Exception as e:
                    time.sleep(0.1)
                    continue
            else:
                # OpenCV VideoCapture
                if self._camera is None or not self._camera.isOpened():
                    time.sleep(0.1)
                    continue
                
                ret, frame = self._camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Store frame
            with self._frame_lock:
                self._frame = frame.copy()
            
            # Process frame
            new_state = self._process_frame(frame)
            
            # Update state
            with self._state_lock:
                self._vision_state = new_state
            
            # Notify callback
            if self._callback:
                try:
                    self._callback(new_state)
                except Exception as e:
                    print(f"[Vision] Callback error: {e}")
            
            # Control frame rate
            time.sleep(0.033)  # ~30 FPS
    
    def _process_frame(self, frame: np.ndarray) -> VisionState:
        """
        Process a single frame and extract vision information.
        
        Args:
            frame: BGR frame from camera
            
        Returns:
            VisionState with detection results
        """
        state = VisionState(timestamp=time.time())
        
        # Get enabled features from config
        vision_config = self.config.get("vision", {})
        
        # Obstacle detection
        if vision_config.get("obstacle_detection", {}).get("enabled", True):
            state.obstacle_ahead, state.obstacle_distance = self._detect_obstacle(frame)
        
        # Face detection
        if vision_config.get("face_detection", {}).get("enabled", True):
            faces = self._detect_faces(frame)
            state.face_detected = len(faces) > 0
            state.face_count = len(faces)
            state.face_positions = faces
        
        # Color detection
        if vision_config.get("color_detection", {}).get("enabled", True):
            detected, pos, radius = self._detect_color(frame)
            state.color_target_detected = detected
            state.color_target_position = pos
            state.color_target_radius = radius
        
        # Motion detection
        if vision_config.get("motion_detection", {}).get("enabled", True):
            detected, area = self._detect_motion(frame)
            state.motion_detected = detected
            state.motion_area = area
        
        return state
    
    def _detect_obstacle(self, frame: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        Detect obstacles in front of the robot.
        
        Uses edge detection and contour analysis to find large objects
        in the center-bottom region of the frame.
        
        Args:
            frame: BGR frame
            
        Returns:
            Tuple of (obstacle_detected, estimated_distance)
        """
        config = self.config.get("vision", {}).get("obstacle_detection", {})
        min_area = config.get("min_area", 5000)
        roi_top = config.get("roi_top", 0.3)
        roi_bottom = config.get("roi_bottom", 0.7)
        
        h, w = frame.shape[:2]
        
        # Define region of interest (center portion of frame)
        roi_y1 = int(h * roi_top)
        roi_y2 = int(h * roi_bottom)
        roi_x1 = int(w * 0.2)
        roi_x2 = int(w * 0.8)
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate to connect edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for large contours (potential obstacles)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Estimate distance based on contour size
                # Larger contour = closer object
                distance = max(0.2, 2.0 - (area / 20000))  # Simple heuristic
                return True, distance
        
        return False, None
    
    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame.
        
        Args:
            frame: BGR frame
            
        Returns:
            List of (x, y, w, h) tuples for each detected face
        """
        if self._face_cascade is None:
            return []
        
        config = self.config.get("vision", {}).get("face_detection", {})
        scale_factor = config.get("scale_factor", 1.2)
        min_neighbors = config.get("min_neighbors", 5)
        min_size = tuple(config.get("min_size", [20, 20]))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return [tuple(face) for face in faces]
    
    def _detect_color(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]], float]:
        """
        Detect target color in frame.
        
        Args:
            frame: BGR frame
            
        Returns:
            Tuple of (detected, (x, y) center position, radius)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for target color
        mask = cv2.inRange(hsv, self._color_lower, self._color_upper)
        
        # Clean up mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, None, 0
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 500:  # Minimum area threshold
            return False, None, 0
        
        # Get enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(largest)
        
        if radius > 10:
            return True, (int(x), int(y)), radius
        
        return False, None, 0
    
    def _detect_motion(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detect motion in frame using background subtraction.
        
        Args:
            frame: BGR frame
            
        Returns:
            Tuple of (motion_detected, (x, y, w, h) bounding box)
        """
        config = self.config.get("vision", {}).get("motion_detection", {})
        min_area = config.get("min_area", 2000)
        threshold = config.get("threshold", 25)
        
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize background model
        if self._motion_avg is None:
            self._motion_avg = gray.copy().astype("float")
            return False, None
        
        # Update background model
        cv2.accumulateWeighted(gray, self._motion_avg, 0.5)
        
        # Compute difference
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self._motion_avg))
        
        # Threshold
        thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                return True, (x, y, w, h)
        
        return False, None
    
    def draw_detections(self, frame: np.ndarray, state: Optional[VisionState] = None) -> np.ndarray:
        """
        Draw detection visualizations on frame.
        
        Args:
            frame: BGR frame to draw on
            state: VisionState to visualize (uses current if None)
            
        Returns:
            Frame with visualizations
        """
        if state is None:
            state = self.get_vision_state()
        
        result = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw obstacle warning
        if state.obstacle_ahead:
            cv2.putText(result, "OBSTACLE AHEAD!", (10, 30), font, 0.7, (0, 0, 255), 2)
            if state.obstacle_distance:
                cv2.putText(result, f"Distance: {state.obstacle_distance:.1f}m", 
                           (10, 60), font, 0.5, (0, 0, 255), 1)
        
        # Draw face rectangles
        for (x, y, w, h) in state.face_positions:
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 128, 64), 2)
        
        if state.face_detected:
            cv2.putText(result, f"Faces: {state.face_count}", (10, 90), font, 0.5, (255, 128, 64), 1)
        
        # Draw color target
        if state.color_target_detected and state.color_target_position:
            x, y = state.color_target_position
            r = int(state.color_target_radius)
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 3, (0, 255, 0), -1)
        
        # Draw motion area
        if state.motion_detected and state.motion_area:
            x, y, w, h = state.motion_area
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(result, "MOTION", (x, y - 10), font, 0.5, (0, 255, 255), 1)
        
        return result


# Factory function
def create_vision_processor(config: Dict[str, Any]) -> VisionProcessor:
    """
    Create a VisionProcessor from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured VisionProcessor instance
    """
    cam_config = config.get("hardware", {}).get("camera", {})
    
    return VisionProcessor(
        camera_id=cam_config.get("device_id", 0),
        width=cam_config.get("width", 640),
        height=cam_config.get("height", 480),
        config=config,
        use_picamera2=cam_config.get("use_picamera2", False)
    )
