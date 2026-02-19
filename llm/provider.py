import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from pydantic import ValidationError

from llm.interfaces import LLMProviderInterface, ToolInterface
from llm.schemas import AgentOutput

class LangChainProvider(LLMProviderInterface):
    """
    Implementation of LLMProviderInterface using LangChain's ChatOpenAI.
    Uses 'with_structured_output' to strictly enforce AgentOutput schema.
    """
    def __init__(self, model_name: str = "meta-llama/llama-3.1-8b-instruct", temperature: float = 0.0):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            # Fallback or error
            pass
            
        self.base_url = "https://openrouter.ai/api/v1"
        self.model_name = model_name
        
        # Initialize the raw LLM
        self.raw_llm = ChatOpenAI(
            model=model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=temperature,
        )

    def generate(self, messages: List[Dict[str, Any]], tools: Optional[List[ToolInterface]] = None) -> AgentOutput:
        """
        Generates a structured response.
        We prompt the model to decide between 'response' or 'tool_call'.
        """
        
        # 1. Convert dict messages to LangChain messages format
        lc_messages = self._convert_messages(messages)
        
        # 2. Bind structured output
        # We force the model to output the AgentOutput Pydantic model
        structured_llm = self.raw_llm.with_structured_output(AgentOutput)
        
        # 3. Augment system prompt with tool definitions if provided
        # (Though structured output handles the schema, providing descriptions helps the model reason)
        if tools:
            tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
            system_update = f"\n\nAvailable Tools:\n{tool_descriptions}\n\nYou must decide whether to use a tool to answer the user's request or respond directly."
            
            # Insert into system prompt if it exists, or add a new one
            if isinstance(lc_messages[0], SystemMessage):
                lc_messages[0].content += system_update
            else:
                lc_messages.insert(0, SystemMessage(content=system_update))

        try:
            # 4. Invoke
            result = structured_llm.invoke(lc_messages)
            return result
        except ValidationError as e:
            # If the model emits raw text instead of JSON, Pydantic validation fails.
            # We can try to salvage the raw content from the exception if possible, 
            # or just assume the last error message contains the content.
            # However, extracting the raw string from the proper place in Langchain's error is hard.
            # A better approach is to wrap the invocation in a way that we can see the raw output.
            
            # Since we can't easily get the raw output from the "invoke" call contributing to the error here without debugging langchain internals,
            # We will try a fallback strategy: if structured output fails, run without it and treat as response.
            print(f"[LLM Schema Error] {e}")
            return self._fallback_generate(lc_messages)
        except Exception as e:
            # Fallback for parsing errors or API errors
            print(f"[LLM Error] {e}")
            return AgentOutput(
                type="response", 
                content="I'm sorry, I encountered an internal error while processing your request."
            )

    def _fallback_generate(self, messages: List[BaseMessage]) -> AgentOutput:
        """Fallback generation without forced structure, assuming response type."""
        try:
            response = self.raw_llm.invoke(messages)
            return AgentOutput(type="response", content=response.content)
        except Exception as e:
             return AgentOutput(type="response", content="Error in fallback generation.")

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        lc_msgs = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "user":
                lc_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_msgs.append(AIMessage(content=content))
            elif role == "tool":
                # For this specific schema, we treated tool results as user/system context or separate message types
                # Simulating tool output as HumanMessage for simplicity called "System" or context
                lc_msgs.append(HumanMessage(content=f"Tool Result: {content}"))
        return lc_msgs
