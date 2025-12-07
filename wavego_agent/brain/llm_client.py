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

Analyze the input and respond with a JSON object. For SIMPLE commands (single action):
{
    "intent": "move|look|stop|query|gesture|light|unknown",
    "action": "specific action name",
    "parameters": {
        "direction": "forward|backward|left|right",
        "duration": 1.0,
        "speed": 100,
        "color": "blue|red|green|yellow|cyan|magenta|cyber|off"
    },
    "reply": "Natural language response to user"
}

For COMPLEX commands (multiple sequential actions like "先前进再后退" or "walk forward then turn left"):
{
    "intent": "sequence",
    "actions": [
        {"action": "forward", "duration": 2.0},
        {"action": "backward", "duration": 2.0}
    ],
    "reply": "Natural language response describing what I'll do"
}

Available actions for sequences:
- Movement: forward, backward, left, right, stop
- Head: look_up, look_down, look_left, look_right
- Gestures: jump, handshake, steady
- Lights: light_red, light_blue, light_green, light_off, etc.
- Pause: wait (use duration to specify seconds)

Examples of complex commands:
- "先向前走2秒，然后向后走" → sequence with [forward 2s, backward 2s]
- "转一圈" → sequence with [left 2s] or [right 2s] (360 degree turn)
- "走个方形" → sequence with [forward, right, forward, right, forward, right, forward]
- "jump then wave" → sequence with [jump, handshake]
- "闪烁红蓝灯" → sequence with [light_red, wait 0.5, light_blue, wait 0.5, light_red...]

Safety rules:
- If vision_state.obstacle_ahead is true, do NOT generate forward movement
- Always acknowledge the user's request in your reply
- For sequences, estimate reasonable durations (default 1-2 seconds per action)

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
        
        # Check for sequence patterns first (multi-step commands)
        sequence_patterns = [
            "then", "and then", "after that", "next",  # English
            "然后", "再", "接着", "之后", "先",  # Chinese
        ]
        
        if any(pattern in msg for pattern in sequence_patterns):
            return self._parse_sequence_command(msg, obstacle_ahead)
        
        # Movement commands
        if any(word in msg for word in ["forward", "ahead", "straight", "前进", "向前"]):
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
        
        if any(word in msg for word in ["backward", "back", "reverse", "后退", "向后"]):
            return {
                "intent": "move",
                "action": "backward",
                "parameters": {"duration": self._extract_duration(msg)},
                "reply": "Moving backward."
            }
        
        if ("left" in msg or "左" in msg) and "look" not in msg and "看" not in msg:
            return {
                "intent": "move",
                "action": "left",
                "parameters": {"duration": self._extract_duration(msg)},
                "reply": "Turning left."
            }
        
        if ("right" in msg or "右" in msg) and "look" not in msg and "看" not in msg:
            return {
                "intent": "move",
                "action": "right",
                "parameters": {"duration": self._extract_duration(msg)},
                "reply": "Turning right."
            }
        
        # Stop commands
        if any(word in msg for word in ["stop", "halt", "freeze", "停", "停止", "别动", "不要动"]):
            return {
                "intent": "stop",
                "action": "stop",
                "parameters": {},
                "reply": "好的，已停止。" if any(c in msg for c in "停别不") else "Stopping."
            }
        
        # Look commands (Chinese)
        if any(phrase in msg for phrase in ["抬头", "向上看", "往上看", "look up"]):
            return {
                "intent": "look",
                "action": "up",
                "parameters": {},
                "reply": "正在抬头。" if any(c in msg for c in "抬上看") else "Looking up."
            }
        
        if any(phrase in msg for phrase in ["低头", "向下看", "往下看", "look down"]):
            return {
                "intent": "look",
                "action": "down",
                "parameters": {},
                "reply": "正在低头。" if any(c in msg for c in "低下看") else "Looking down."
            }
        
        if any(phrase in msg for phrase in ["向左看", "往左看", "左转头", "look left"]):
            return {
                "intent": "look",
                "action": "left",
                "parameters": {},
                "reply": "正在向左看。" if any(c in msg for c in "左看转") else "Looking left."
            }
        
        if any(phrase in msg for phrase in ["向右看", "往右看", "右转头", "look right"]):
            return {
                "intent": "look",
                "action": "right",
                "parameters": {},
                "reply": "正在向右看。" if any(c in msg for c in "右看转") else "Looking right."
            }
        
        # Gestures
        if any(word in msg for word in ["jump", "跳", "跳跃"]):
            return {
                "intent": "gesture",
                "action": "jump",
                "parameters": {},
                "reply": "跳！" if any(c in msg for c in "跳") else "Jumping!"
            }
        
        if any(word in msg for word in ["shake", "hand", "hello", "hi", "握手", "打招呼", "你好"]):
            return {
                "intent": "gesture",
                "action": "handshake",
                "parameters": {},
                "reply": "你好！很高兴见到你！" if any(c in msg for c in "握招你") else "Nice to meet you!"
            }
        
        if any(word in msg for word in ["steady", "balance", "平衡", "稳定"]):
            return {
                "intent": "gesture",
                "action": "steady",
                "parameters": {},
                "reply": "正在启动平衡模式。" if any(c in msg for c in "平稳") else "Activating steady mode."
            }
        
        # Light commands (English and Chinese)
        color_map = {
            "blue": "blue", "蓝": "blue", "蓝色": "blue",
            "red": "red", "红": "red", "红色": "red",
            "green": "green", "绿": "green", "绿色": "green",
            "yellow": "yellow", "黄": "yellow", "黄色": "yellow",
            "cyan": "cyan", "青": "cyan", "青色": "cyan",
            "magenta": "magenta", "紫": "magenta", "紫色": "magenta",
            "cyber": "cyber", "白": "cyber", "白色": "cyber",
        }
        
        for color_word, color_value in color_map.items():
            if color_word in msg:
                is_chinese = any(ord(c) > 127 for c in msg)
                return {
                    "intent": "light",
                    "action": "color",
                    "parameters": {"color": color_value},
                    "reply": f"灯光设置为{color_word}色。" if is_chinese else f"Setting lights to {color_value}."
                }
        
        if any(phrase in msg for phrase in ["light off", "lights off", "turn off light", "关灯", "灯关"]):
            is_chinese = any(ord(c) > 127 for c in msg)
            return {
                "intent": "light",
                "action": "off",
                "parameters": {"color": "off"},
                "reply": "已关闭灯光。" if is_chinese else "Turning off lights."
            }
        
        # Query commands
        query_words_en = ["what", "where", "how", "status", "doing", "see"]
        query_words_zh = ["什么", "哪里", "怎么", "状态", "干什么", "看到", "在做", "你好吗"]
        
        if any(word in msg for word in query_words_en + query_words_zh):
            is_chinese = any(ord(c) > 127 for c in msg)
            return {
                "intent": "query",
                "action": "status",
                "parameters": {},
                "reply": "我已准备好，等待你的命令。你可以说'前进'、'后退'、'跳'、'握手'等。" if is_chinese else "I'm ready and waiting for your commands."
            }
        
        if any(word in msg for word in ["obstacle", "障碍", "前面"]):
            is_chinese = any(ord(c) > 127 for c in msg)
            if obstacle_ahead:
                return {
                    "intent": "query",
                    "action": "obstacle_check",
                    "parameters": {},
                    "reply": "是的，前方检测到障碍物。" if is_chinese else "Yes, I detect an obstacle ahead."
                }
            else:
                return {
                    "intent": "query",
                    "action": "obstacle_check",
                    "parameters": {},
                    "reply": "前方没有检测到障碍物。" if is_chinese else "No obstacles detected ahead."
                }
        
        # Default unknown
        is_chinese = any(ord(c) > 127 for c in msg)
        if is_chinese:
            return {
                "intent": "unknown",
                "action": "none",
                "parameters": {},
                "reply": "抱歉，我不太理解。你可以试试这些命令：'前进'、'后退'、'左转'、'右转'、'跳'、'握手'、'抬头'、'红色灯'。"
            }
        else:
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
    
    API key can be set in two ways:
    1. Directly in config: llm.api_key: "sk-xxx"
    2. Via environment variable: llm.api_key_env: "OPENAI_API_KEY"
    
    Args:
        config: Configuration dictionary
        use_mock: If True, create a mock client for testing
        
    Returns:
        Configured LLMClient instance
    """
    if use_mock:
        return MockLLMClient()
    
    llm_config = config.get("llm", {})
    
    # Try to get API key in order of priority:
    # 1. Direct api_key in config
    # 2. Environment variable specified in api_key_env
    # 3. Default OPENAI_API_KEY environment variable
    api_key = llm_config.get("api_key")
    if not api_key:
        env_var = llm_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(env_var)
    
    return LLMClient(
        api_key=api_key,
        model=llm_config.get("model", "gpt-4o-mini"),
        base_url=llm_config.get("base_url"),  # Support custom endpoints
        max_tokens=llm_config.get("max_tokens", 500),
        temperature=llm_config.get("temperature", 0.7),
        system_prompt=llm_config.get("system_prompt")
    )
