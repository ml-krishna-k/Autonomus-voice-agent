from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

class AgentOutput(BaseModel):
    """
    Unified schema for LLM output.
    Enforces the structure required for the execution engine.
    """
    type: Literal["response", "tool_call"] = Field(
        ..., 
        description="The type of the output. 'response' for final answer to user, 'tool_call' if an action is needed."
    )
    content: str = Field(
        ..., 
        description="The natural language content. For 'response', this is the message to users. For 'tool_call', this is the reasoning/thought."
    )
    tool_name: Optional[str] = Field(
        None, 
        description="The name of the tool to call. Required if type is 'tool_call'."
    )
    arguments: Optional[Dict[str, Any]] = Field(
        None, 
        description="The arguments for the tool call. Required if type is 'tool_call'."
    )
