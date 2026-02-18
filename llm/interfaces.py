from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from llm.schemas import AgentOutput

class ToolInterface(ABC):
    """Abstract base class for tools."""
    name: str
    description: str
    args_schema: Any

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        pass

class LLMProviderInterface(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], tools: Optional[List[ToolInterface]] = None) -> AgentOutput:
        """
        Generates a structured response from the LLM.
        Must return an AgentOutput object.
        """
        pass
