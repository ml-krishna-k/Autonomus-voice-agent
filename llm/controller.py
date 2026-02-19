import time
import logging
from typing import List, Dict, Any, Generator

from llm.interfaces import LLMProviderInterface
from llm.registry import ToolRegistry
from llm.schemas import AgentOutput

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutionController:
    """
    Explicit execution controller for the agent.
    Orchestrates the loop: LLM -> Parse -> Execute -> Resume.
    """
    def __init__(self, provider: LLMProviderInterface, registry: ToolRegistry):
        self.provider = provider
        self.registry = registry
        self.max_turns = 5

    def run(self, user_input: str, history: List[Dict[str, Any]], system_prompt: str) -> Generator[str, None, None]:
        """
        Main execution pipeline.
        Yields strings for TTS streaming (content only).
        """
        
        # 1. Prepare initial context
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]
        
        turn = 0
        while turn < self.max_turns:
            turn += 1
            # logger.info(f"--- Turn {turn} ---")
            
            # 2. Call LLM
            start_time = time.time()
            output: AgentOutput = self.provider.generate(messages, tools=self.registry.list_tools())
            latency = time.time() - start_time
                # logger.info(f"LLM Latency: {latency:.2f}s | Type: {output.type}")

            # 3. Handle 'response' (Final Answer)
            if output.type == "response":
                # We yield the content for the user
                yield output.content
                
                # Append to history and break
                messages.append({"role": "assistant", "content": output.content})
                break

            # 4. Handle 'tool_call'
            elif output.type == "tool_call":
                tool_name = output.tool_name
                args = output.arguments or {}
                
                # logger.info(f"Tool Call: {tool_name} with args {args}")
                yield f"I'm checking {tool_name}..." # Optional: filler words for latency masking

                # Execute Tool
                tool = self.registry.get_tool(tool_name)
                if tool:
                    try:
                        result = tool.execute(**args)
                        result_str = str(result)
                    except Exception as e:
                        result_str = f"Error executing tool: {e}"
                else:
                    result_str = f"Error: Tool {tool_name} not found."

                # logger.info(f"Tool Result: {result_str}")

                # Update history with the intermediary steps
                # Note: We append the assistant's "thought" (content) and the tool result
                # This helps the LLM know what it just did.
                messages.append({"role": "assistant", "content": f"Thought: {output.content}\nCall: {tool_name}({args})"})
                messages.append({"role": "tool", "content": result_str})
                
                # Continue loop -> LLM will see result and formulate response
                continue
        
        if turn >= self.max_turns:
             logger.warning("Max turns reached.")
             yield "I'm sorry, I couldn't complete the task in time."

    def get_updated_history(self, original_history: List[Dict[str, Any]], final_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Helper to extract new messages to save.
        Implementation depends on how specific the history storage format is.
        """
        # This is a placeholder for logic to diff `final_messages` against `original_history`
        # In a real app, you'd probably just append the new turns.
        return final_messages
