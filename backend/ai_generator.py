import logging
import anthropic
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **search_course_content**: Use **only** for questions about specific course content or detailed educational materials
- **get_course_outline**: Use for questions about a course outline, structure or lesson list
- **Up to 2 sequential tool rounds per query**: after each round you see the results and can decide whether another call is needed
- Use a second round only when it depends on the first round's results (e.g. get an outline to learn a lesson's title, then search for that topic in other courses)
- Make independent lookups together in the same round
- Answer as soon as you have enough information
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results or an error, retry once with corrected input or state this clearly without offering alternatives

Outline Responses:
- Return the course title, the course link, and every lesson with its number and title

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Get the outline first, then answer
- **Course-specific questions**: Search first, then answer
- **Multi-step questions**: Look up what the next step depends on, then do the dependent lookup, then give one final answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        params = {**self.base_params, "system": system_content}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = {"type": "auto"}

        messages = [{"role": "user", "content": query}]
        response = self._request(params, messages)

        # stop_reason, not block presence: a max_tokens cut can leave a truncated tool_use
        rounds = 0
        while response.stop_reason == "tool_use" and tools and tool_manager:
            messages.append({"role": "assistant", "content": response.content})
            results, failed = self._execute_tools(response, tool_manager)
            messages.append({"role": "user", "content": results})
            rounds += 1

            final = failed or rounds >= self.max_tool_rounds
            response = self._request(params, messages, allow_tools=not final)
            if final:
                break

        return self._extract_text(response)

    def _request(self, params: Dict[str, Any], messages: List, allow_tools: bool = True):
        """Call Claude; without allow_tools, keep tools but forbid calling them"""
        call = {**params, "messages": list(messages)}
        # Keep tools and system identical so returned thinking blocks stay valid
        if not allow_tools:
            call["tool_choice"] = {"type": "none"}
        return self.client.messages.create(**call)

    @staticmethod
    def _execute_tools(response, tool_manager):
        """Run every tool_use block; return (tool_results, failed)"""
        results, failed = [], False
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = {"type": "tool_result", "tool_use_id": block.id}
            try:
                result["content"] = tool_manager.execute_tool(block.name, **block.input)
            except Exception as e:
                logger.exception("Tool %s failed", block.name)
                result["content"] = f"Tool '{block.name}' failed: {e}"
                result["is_error"] = True
                failed = True
            results.append(result)
        return results, failed

    @staticmethod
    def _extract_text(response) -> str:
        """Join text blocks, skipping thinking and tool_use blocks"""
        return "".join(block.text for block in response.content if block.type == "text")