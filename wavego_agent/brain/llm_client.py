"""
LLM Client Module
=================
Handles communication with Large Language Model APIs.
Supports OpenAI API and compatible endpoints.
"""

import os
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """
    Response from LLM API.
    
    Attributes:
        text: Raw text response
        parsed: Parsed JSON object (if applicable)
        success: Whether the call succeeded
        error: Error message if failed
        tokens_used: Number of tokens used
        latency: Response time in seconds
    """
    text: str = ""
    parsed: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0
    latency: float = 0.0


class LLMClient:
    """
    Client for LLM API communication.
    
    Provides a unified interface for sending prompts and receiving
    structured responses from language models.
    
    Example:
        client = LLMClient(api_key="sk-...", model="gpt-4o-mini")
        
        response = client.chat(
            user_message="Move forward 1 meter",
            system_prompt="You are a robot controller...",
            context={"vision_state": {...}}
        )
        
        if response.success and response.parsed:
            intent = response.parsed.get("intent")
    """
    
    # Default system prompt for robot control
    DEFAULT_SYSTEM_PROMPT = """You are an intelligent assistant controlling a quadruped robot dog called WaveGo.
You receive user commands (voice transcriptions) and vision state information.

Analyze the input and respond with a JSON object containing:
{
    "intent": "move|look|stop|query|gesture|light|unknown",
    "action": "specific action name",
    "parameters": {
        "direction": "forward|backward|left|right",
        "duration": 1.0,
        "speed": 100,
        "color": "blue|red|green|yellow|cyan|magenta|off"
    },
    "reply": "Natural language response to user"
}

Available actions:
- Movement (intent="move"): forward, backward, left, right, stop
- Head control (intent="look"): up, down, left, right, stop
- Gestures (intent="gesture"): jump, handshake, steady
- Lights (intent="light"): on, off, color
- Stop (intent="stop"): stop all movement

Safety rules:
- If vision_state.obstacle_ahead is true, do NOT generate forward movement
- Always acknowledge the user's request in your reply

IMPORTANT: Always respond with valid JSON only, no additional text."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key (or uses OPENAI_API_KEY env var)
            model: Model name to use
            base_url: Custom API base URL (for compatible endpoints)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            system_prompt: Custom system prompt
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        self._client = None
        self._init_client()
        
        # Conversation history for context
        self._history: List[Dict[str, str]] = []
        self._max_history = 10
    
    def _init_client(self):
        """Initialize the OpenAI client."""
        if not self.api_key:
            print("[LLM] Warning: No API key provided. Set OPENAI_API_KEY environment variable.")
            return
        
        try:
            from openai import OpenAI
            
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            
            self._client = OpenAI(**kwargs)
            print(f"[LLM] Initialized with model: {self.model}")
            
        except ImportError:
            print("[LLM] openai package not installed. Install with: pip install openai")
        except Exception as e:
            print(f"[LLM] Failed to initialize client: {e}")
    
    def is_available(self) -> bool:
        """Check if the LLM client is available."""
        return self._client is not None
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        include_history: bool = True,
        parse_json: bool = True
    ) -> LLMResponse:
        """
        Send a chat message and get a response.
        
        Args:
            user_message: The user's message/command
            system_prompt: Override system prompt (uses default if None)
            context: Additional context (vision_state, robot_state, etc.)
            include_history: Whether to include conversation history
            parse_json: Whether to attempt JSON parsing of response
            
        Returns:
            LLMResponse with text and optionally parsed JSON
        """
        if not self.is_available():
            return LLMResponse(
                success=False,
                error="LLM client not available"
            )
        
        start_time = time.time()
        
        # Build messages
        messages = []
        
        # System message
        sys_prompt = system_prompt or self.system_prompt
        messages.append({"role": "system", "content": sys_prompt})
        
        # Add history if enabled
        if include_history and self._history:
            messages.extend(self._history[-self._max_history:])
        
        # Build user message with context
        full_user_message = user_message
        if context:
            context_str = json.dumps(context, indent=2)
            full_user_message = f"Context:\n{context_str}\n\nUser command: {user_message}"
        
        messages.append({"role": "user", "content": full_user_message})
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            latency = time.time() - start_time
            
            # Extract response text
            text = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0
            
            # Update history
            self._history.append({"role": "user", "content": user_message})
            self._history.append({"role": "assistant", "content": text})
            
            # Trim history if too long
            if len(self._history) > self._max_history * 2:
                self._history = self._history[-self._max_history * 2:]
            
            # Try to parse JSON
            parsed = None
            if parse_json:
                parsed = self._parse_json_response(text)
            
            return LLMResponse(
                text=text,
                parsed=parsed,
                success=True,
                tokens_used=tokens,
                latency=latency
            )
            
        except Exception as e:
            return LLMResponse(
                success=False,
                error=str(e),
                latency=time.time() - start_time
            )
    
    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse JSON from LLM response.
        
        Args:
            text: Raw response text
            
        Returns:
            Parsed dict or None if parsing fails
        """
        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in response (handle markdown code blocks)
        import re
        
        # Look for ```json ... ``` blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Look for { ... } blocks
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def clear_history(self):
        """Clear conversation history."""
        self._history = []
    
    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without API calls.
    
    Provides simple pattern-matching responses for basic commands.
    """
    
    def __init__(self, **kwargs):
        """Initialize mock client (ignores API key)."""
        self.model = "mock"
        self._client = True  # Pretend we have a client
        self._history = []
        self._max_history = 10
        self.system_prompt = kwargs.get("system_prompt", "")
    
    def is_available(self) -> bool:
        return True
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        include_history: bool = True,
        parse_json: bool = True
    ) -> LLMResponse:
        """Generate mock response based on simple pattern matching."""
        
        msg_lower = user_message.lower()
        
        # Check for obstacle if context provided
        obstacle_ahead = False
        if context and context.get("vision_state", {}).get("obstacle_ahead"):
            obstacle_ahead = True
        
        # Simple pattern matching
        response_data = self._match_command(msg_lower, obstacle_ahead)
        
        return LLMResponse(
            text=json.dumps(response_data),
            parsed=response_data,
            success=True,
            tokens_used=0,
            latency=0.01
        )
    
    def _match_command(self, msg: str, obstacle_ahead: bool) -> Dict[str, Any]:
        """Match command patterns and generate response."""
        
        # Movement commands
        if any(word in msg for word in ["forward", "ahead", "straight"]):
            if obstacle_ahead:
                return {
                    "intent": "stop",
                    "action": "stop",
                    "parameters": {},
                    "reply": "I cannot move forward, there's an obstacle ahead."
                }
            return {
                "intent": "move",
                "action": "forward",
                "parameters": {"duration": self._extract_duration(msg)},
                "reply": "Moving forward."
            }
        
        if any(word in msg for word in ["backward", "back", "reverse"]):
            return {
                "intent": "move",
                "action": "backward",
                "parameters": {"duration": self._extract_duration(msg)},
                "reply": "Moving backward."
            }
        
        if "left" in msg and "look" not in msg:
            return {
                "intent": "move",
                "action": "left",
                "parameters": {"duration": self._extract_duration(msg)},
                "reply": "Turning left."
            }
        
        if "right" in msg and "look" not in msg:
            return {
                "intent": "move",
                "action": "right",
                "parameters": {"duration": self._extract_duration(msg)},
                "reply": "Turning right."
            }
        
        # Stop commands
        if any(word in msg for word in ["stop", "halt", "freeze"]):
            return {
                "intent": "stop",
                "action": "stop",
                "parameters": {},
                "reply": "Stopping."
            }
        
        # Look commands
        if "look up" in msg:
            return {
                "intent": "look",
                "action": "up",
                "parameters": {},
                "reply": "Looking up."
            }
        
        if "look down" in msg:
            return {
                "intent": "look",
                "action": "down",
                "parameters": {},
                "reply": "Looking down."
            }
        
        if "look left" in msg:
            return {
                "intent": "look",
                "action": "left",
                "parameters": {},
                "reply": "Looking left."
            }
        
        if "look right" in msg:
            return {
                "intent": "look",
                "action": "right",
                "parameters": {},
                "reply": "Looking right."
            }
        
        # Gestures
        if "jump" in msg:
            return {
                "intent": "gesture",
                "action": "jump",
                "parameters": {},
                "reply": "Jumping!"
            }
        
        if any(word in msg for word in ["shake", "hand", "hello", "hi"]):
            return {
                "intent": "gesture",
                "action": "handshake",
                "parameters": {},
                "reply": "Nice to meet you!"
            }
        
        if "steady" in msg or "balance" in msg:
            return {
                "intent": "gesture",
                "action": "steady",
                "parameters": {},
                "reply": "Activating steady mode."
            }
        
        # Light commands
        for color in ["blue", "red", "green", "yellow", "cyan", "magenta"]:
            if color in msg:
                return {
                    "intent": "light",
                    "action": "color",
                    "parameters": {"color": color},
                    "reply": f"Setting lights to {color}."
                }
        
        if "light" in msg and ("off" in msg or "turn off" in msg):
            return {
                "intent": "light",
                "action": "off",
                "parameters": {"color": "off"},
                "reply": "Turning off lights."
            }
        
        # Query commands
        if any(word in msg for word in ["what", "where", "how", "status", "doing"]):
            return {
                "intent": "query",
                "action": "status",
                "parameters": {},
                "reply": "I'm ready and waiting for your commands."
            }
        
        if "obstacle" in msg:
            if obstacle_ahead:
                return {
                    "intent": "query",
                    "action": "obstacle_check",
                    "parameters": {},
                    "reply": "Yes, I detect an obstacle ahead."
                }
            else:
                return {
                    "intent": "query",
                    "action": "obstacle_check",
                    "parameters": {},
                    "reply": "No obstacles detected ahead."
                }
        
        # Default unknown
        return {
            "intent": "unknown",
            "action": "none",
            "parameters": {},
            "reply": "I'm not sure what you want me to do. Try commands like 'move forward', 'turn left', 'jump', or 'look up'."
        }
    
    def _extract_duration(self, msg: str) -> float:
        """Extract duration from message."""
        import re
        
        # Look for patterns like "1 second", "2 seconds", "1.5 sec"
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:second|sec|s)', msg)
        if match:
            return float(match.group(1))
        
        # Look for patterns like "half meter", "1 meter"
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:meter|m)', msg)
        if match:
            # Assume ~1 second per 0.5 meters
            return float(match.group(1)) * 2
        
        if "half" in msg:
            return 1.0
        
        return 1.0  # Default duration


# Factory function
def create_llm_client(config: Dict[str, Any], use_mock: bool = False) -> LLMClient:
    """
    Create an LLM client from configuration.
    
    Args:
        config: Configuration dictionary
        use_mock: If True, create a mock client for testing
        
    Returns:
        Configured LLMClient instance
    """
    if use_mock:
        return MockLLMClient()
    
    llm_config = config.get("llm", {})
    
    api_key = os.environ.get(llm_config.get("api_key_env", "OPENAI_API_KEY"))
    
    return LLMClient(
        api_key=api_key,
        model=llm_config.get("model", "gpt-4o-mini"),
        max_tokens=llm_config.get("max_tokens", 500),
        temperature=llm_config.get("temperature", 0.7),
        system_prompt=llm_config.get("system_prompt")
    )
