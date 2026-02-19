from typing import Dict, Type, Callable, Any, List, Optional
from llm.interfaces import ToolInterface
from pydantic import BaseModel

class FunctionalTool(ToolInterface):
    """
    A concrete implementation of ToolInterface that wraps a python function.
    """
    def __init__(self, name: str, func: Callable, description: str, args_schema: Type[BaseModel]):
        self.name = name
        self.func = func
        self.description = description
        self.args_schema = args_schema

    def execute(self, **kwargs) -> Any:
        # Validate arguments using Pydantic schema
        # args = self.args_schema(**kwargs) # Optional validation if we trust the LLM's structured output enough, but safer to validate.
        return self.func(**kwargs)

class ToolRegistry:
    """
    Central repository for available tools.
    """
    def __init__(self):
        self._tools: Dict[str, ToolInterface] = {}

    def register(self, tool: ToolInterface):
        """Register a tool instance."""
        if tool.name in self._tools:
            print(f"[WARN] Overwriting tool: {tool.name}")
        self._tools[tool.name] = tool

    def register_function(self, name: str, description: str, args_schema: Type[BaseModel]):
        """Decorator to register a function as a tool."""
        def decorator(func):
            tool = FunctionalTool(name, func, description, args_schema)
            self.register(tool)
            return func
        return decorator

    def get_tool(self, name: str) -> Optional[ToolInterface]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolInterface]:
        return list(self._tools.values())
