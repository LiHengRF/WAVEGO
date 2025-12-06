"""
Agent Module
============
The "Brain" of the robot - handles reasoning and decision making.
Bridges user intent (from LLM) with motor commands.
"""

import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.command_types import (
    Intent, MoveAction, LookAction, GestureAction, LightColor,
    MotorCommand, VisionState, RobotState, AgentDecision,
    create_move_command, create_look_command,
    create_gesture_command, create_light_command, create_buzzer_command
)
from core.robot_state import RobotStateManager, get_state_manager
from brain.llm_client import LLMClient, LLMResponse


class Agent:
    """
    Main Agent class for robot decision making.
    
    The Agent is responsible for:
    1. Receiving user commands (text) and vision state
    2. Using LLM for high-level intent understanding
    3. Applying safety constraints
    4. Generating motor commands
    5. Producing user-friendly responses
    
    Example:
        agent = Agent(llm_client, state_manager)
        
        decision = agent.decide(
            user_text="Move forward half a meter",
            vision_state=vision_processor.get_vision_state()
        )
        
        for cmd in decision.motor_commands:
            motor_controller.execute(cmd)
        
        tts.speak(decision.reply_text)
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        state_manager: Optional[RobotStateManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Agent.
        
        Args:
            llm_client: LLM client for language understanding
            state_manager: Robot state manager (creates default if None)
            config: Agent configuration dictionary
        """
        self.llm = llm_client
        self.state = state_manager or get_state_manager()
        self.config = config or {}
        
        # Safety settings
        safety_config = self.config.get("agent", {}).get("safety", {})
        self.obstacle_stop = safety_config.get("obstacle_stop", True)
        self.max_move_time = safety_config.get("max_continuous_move_time", 5.0)
        
        # Movement settings
        move_config = self.config.get("agent", {}).get("movement", {})
        self.default_speed = move_config.get("default_speed", 100)
        self.default_duration = move_config.get("default_duration", 1.0)
    
    def decide(
        self,
        user_text: str,
        vision_state: Optional[VisionState] = None,
        bypass_llm: bool = False
    ) -> AgentDecision:
        """
        Make a decision based on user input and current state.
        
        Args:
            user_text: User's command (from STT)
            vision_state: Current vision state (from camera)
            bypass_llm: If True, use simple pattern matching instead of LLM
            
        Returns:
            AgentDecision with motor commands and reply text
        """
        # Get current states
        robot_state = self.state.robot_state
        if vision_state is None:
            vision_state = self.state.vision_state
        
        # Build context for LLM
        context = {
            "robot_state": robot_state.to_dict(),
            "vision_state": vision_state.to_dict()
        }
        
        # Get LLM response
        if bypass_llm:
            # Use simple pattern matching (useful for testing)
            llm_response = self._simple_intent_parse(user_text, context)
        else:
            llm_response = self.llm.chat(
                user_message=user_text,
                context=context
            )
        
        if not llm_response.success:
            return AgentDecision(
                intent=Intent.UNKNOWN,
                reply_text=f"I had trouble understanding. Error: {llm_response.error}",
                success=False,
                error_message=llm_response.error
            )
        
        # Parse LLM response into structured decision
        decision = self._parse_llm_response(llm_response, vision_state, robot_state)
        
        return decision
    
    def _parse_llm_response(
        self,
        response: LLMResponse,
        vision_state: VisionState,
        robot_state: RobotState
    ) -> AgentDecision:
        """
        Parse LLM response into agent decision with safety checks.
        
        Args:
            response: LLM response
            vision_state: Current vision state
            robot_state: Current robot state
            
        Returns:
            AgentDecision with motor commands
        """
        # Default decision
        decision = AgentDecision(
            intent=Intent.UNKNOWN,
            reply_text="I'm not sure what to do."
        )
        
        # Get parsed data from LLM
        data = response.parsed
        if not data:
            decision.reply_text = response.text or "I couldn't parse the command."
            return decision
        
        # Extract intent
        intent_str = data.get("intent", "unknown").lower()
        try:
            decision.intent = Intent(intent_str)
        except ValueError:
            decision.intent = Intent.UNKNOWN
        
        # Extract action and parameters
        action = data.get("action", "").lower()
        params = data.get("parameters", {})
        reply = data.get("reply", "")
        
        # Generate motor commands based on intent
        commands, modified_reply = self._generate_commands(
            decision.intent, action, params, vision_state, robot_state
        )
        
        decision.motor_commands = commands
        decision.reply_text = modified_reply if modified_reply else reply
        
        return decision
    
    def _generate_commands(
        self,
        intent: Intent,
        action: str,
        params: Dict[str, Any],
        vision_state: VisionState,
        robot_state: RobotState
    ) -> Tuple[List[MotorCommand], Optional[str]]:
        """
        Generate motor commands with safety checks.
        
        Args:
            intent: Detected intent
            action: Specific action
            params: Action parameters
            vision_state: Current vision state
            robot_state: Current robot state
            
        Returns:
            Tuple of (commands list, optional modified reply)
        """
        commands = []
        modified_reply = None
        
        # Get duration/speed from params or use defaults
        duration = params.get("duration", self.default_duration)
        duration = min(duration, self.max_move_time)  # Safety limit
        
        if intent == Intent.MOVE:
            # Safety check: obstacle ahead
            if action == "forward" and vision_state.obstacle_ahead and self.obstacle_stop:
                modified_reply = "I can't move forward - there's an obstacle ahead!"
                return [], modified_reply
            
            commands, modified_reply = self._handle_move(action, duration)
            
        elif intent == Intent.LOOK:
            commands = self._handle_look(action)
            
        elif intent == Intent.STOP:
            commands = self._handle_stop()
            
        elif intent == Intent.GESTURE:
            commands = self._handle_gesture(action)
            
        elif intent == Intent.LIGHT:
            color = params.get("color", "blue")
            commands = self._handle_light(action, color)
            
        elif intent == Intent.QUERY:
            # Query doesn't generate motor commands
            pass
        
        return commands, modified_reply
    
    def _handle_move(self, action: str, duration: float) -> Tuple[List[MotorCommand], Optional[str]]:
        """Generate movement commands."""
        commands = []
        modified_reply = None
        
        action_map = {
            "forward": MoveAction.FORWARD,
            "backward": MoveAction.BACKWARD,
            "left": MoveAction.LEFT,
            "right": MoveAction.RIGHT,
            "stop": MoveAction.STOP,
        }
        
        move_action = action_map.get(action)
        if move_action:
            if move_action in [MoveAction.FORWARD, MoveAction.BACKWARD]:
                # Start movement
                commands.append(create_move_command(move_action, duration))
                # Add stop command after duration
                commands.append(create_move_command(MoveAction.STOP_FB, None))
            elif move_action in [MoveAction.LEFT, MoveAction.RIGHT]:
                commands.append(create_move_command(move_action, duration))
                commands.append(create_move_command(MoveAction.STOP_LR, None))
            else:
                commands.append(create_move_command(MoveAction.STOP_FB))
                commands.append(create_move_command(MoveAction.STOP_LR))
        
        return commands, modified_reply
    
    def _handle_look(self, action: str) -> List[MotorCommand]:
        """Generate head/look commands."""
        action_map = {
            "up": LookAction.UP,
            "down": LookAction.DOWN,
            "left": LookAction.LEFT,
            "right": LookAction.RIGHT,
            "stop": LookAction.STOP_UD,
        }
        
        look_action = action_map.get(action)
        if look_action:
            return [create_look_command(look_action, 0.5)]
        
        return []
    
    def _handle_stop(self) -> List[MotorCommand]:
        """Generate stop commands."""
        return [
            create_move_command(MoveAction.STOP_FB),
            create_move_command(MoveAction.STOP_LR),
            create_look_command(LookAction.STOP_UD),
            create_look_command(LookAction.STOP_LR),
        ]
    
    def _handle_gesture(self, action: str) -> List[MotorCommand]:
        """Generate gesture commands."""
        action_map = {
            "jump": GestureAction.JUMP,
            "handshake": GestureAction.HANDSHAKE,
            "steady": GestureAction.STEADY,
        }
        
        gesture_action = action_map.get(action)
        if gesture_action:
            return [create_gesture_command(gesture_action)]
        
        return []
    
    def _handle_light(self, action: str, color: str) -> List[MotorCommand]:
        """Generate light commands."""
        color_map = {
            "off": LightColor.OFF,
            "blue": LightColor.BLUE,
            "red": LightColor.RED,
            "green": LightColor.GREEN,
            "yellow": LightColor.YELLOW,
            "cyan": LightColor.CYAN,
            "magenta": LightColor.MAGENTA,
            "cyber": LightColor.CYBER,
        }
        
        light_color = color_map.get(color.lower(), LightColor.BLUE)
        return [create_light_command(light_color)]
    
    def _simple_intent_parse(self, text: str, context: Dict) -> LLMResponse:
        """
        Simple pattern-matching fallback for intent parsing.
        Used when LLM is unavailable or for testing.
        """
        from brain.llm_client import MockLLMClient
        mock = MockLLMClient()
        return mock.chat(text, context=context)
    
    def get_status_response(self) -> str:
        """
        Generate a status response for query intents.
        
        Returns:
            Human-readable status string
        """
        robot_state = self.state.robot_state
        vision_state = self.state.vision_state
        
        status_parts = []
        
        # Mode
        status_parts.append(f"I'm in {robot_state.mode} mode.")
        
        # Movement
        if robot_state.is_moving:
            status_parts.append("I'm currently moving.")
        else:
            status_parts.append("I'm standing still.")
        
        # Last action
        if robot_state.last_action != "none":
            status_parts.append(f"My last action was: {robot_state.last_action}.")
        
        # Vision
        if vision_state.obstacle_ahead:
            status_parts.append("I see an obstacle ahead.")
        
        if vision_state.face_detected:
            status_parts.append(f"I can see {vision_state.face_count} face(s).")
        
        if vision_state.motion_detected:
            status_parts.append("I detect movement around me.")
        
        return " ".join(status_parts)


class AgentLoop:
    """
    Main loop for continuous agent operation.
    
    Coordinates STT, vision, agent decision, motor control, and TTS
    in a continuous loop.
    
    Example:
        loop = AgentLoop(agent, stt, vision, motor, tts)
        loop.start()
        
        # Later...
        loop.stop()
    """
    
    def __init__(
        self,
        agent: Agent,
        stt,  # SpeechToText
        vision,  # VisionProcessor
        motor,  # MotorController
        tts,  # TextToSpeech
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agent loop.
        
        Args:
            agent: Agent instance
            stt: Speech-to-text instance
            vision: Vision processor instance
            motor: Motor controller instance
            tts: Text-to-speech instance
            config: Configuration dictionary
        """
        self.agent = agent
        self.stt = stt
        self.vision = vision
        self.motor = motor
        self.tts = tts
        self.config = config or {}
        
        self._is_running = False
        self._thread = None
    
    def start(self):
        """Start the agent loop."""
        if self._is_running:
            print("[AgentLoop] Already running")
            return
        
        self._is_running = True
        
        # Start continuous STT with callback
        self.stt.start_continuous(callback=self._on_speech)
        
        print("[AgentLoop] Started - listening for commands...")
    
    def stop(self):
        """Stop the agent loop."""
        self._is_running = False
        self.stt.stop_continuous()
        print("[AgentLoop] Stopped")
    
    def _on_speech(self, text: str):
        """
        Callback for when speech is recognized.
        
        Args:
            text: Recognized speech text
        """
        if not text or not self._is_running:
            return
        
        print(f"[AgentLoop] Heard: {text}")
        
        # Get current vision state
        vision_state = self.vision.get_vision_state()
        
        # Make decision
        decision = self.agent.decide(text, vision_state)
        
        print(f"[AgentLoop] Decision: intent={decision.intent.value}, "
              f"commands={len(decision.motor_commands)}")
        
        # Speak response (before or after action based on config)
        speak_first = self.config.get("agent", {}).get("response", {}).get("speak_confirmation", True)
        
        if speak_first and decision.reply_text:
            self.tts.speak(decision.reply_text)
        
        # Execute motor commands
        for cmd in decision.motor_commands:
            self.motor.execute(cmd)
        
        # Speak after if configured
        if not speak_first and decision.reply_text:
            self.tts.speak(decision.reply_text)
        
        # Update robot state
        if decision.motor_commands:
            action_desc = f"{decision.intent.value}: {len(decision.motor_commands)} commands"
            self.agent.state.update_action(action_desc)
    
    def process_single_command(self, text: str) -> AgentDecision:
        """
        Process a single command (useful for testing).
        
        Args:
            text: Command text
            
        Returns:
            AgentDecision
        """
        vision_state = self.vision.get_vision_state()
        decision = self.agent.decide(text, vision_state)
        
        # Execute
        for cmd in decision.motor_commands:
            self.motor.execute(cmd)
        
        if decision.reply_text:
            self.tts.speak(decision.reply_text)
        
        return decision


# Factory function
def create_agent(llm_client: LLMClient, config: Dict[str, Any]) -> Agent:
    """
    Create an Agent from configuration.
    
    Args:
        llm_client: LLM client instance
        config: Configuration dictionary
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        llm_client=llm_client,
        config=config
    )
