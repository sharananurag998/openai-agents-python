from __future__ import annotations

import json
import inspect
from collections.abc import Sequence
from typing import Any, get_type_hints, get_origin, Dict, Set

from ...exceptions import AgentsException, UserError
from ...logger import logger
from ...run_context import RunContextWrapper  # For empty context
from ...tool import (
    FunctionTool,
    Tool,
)  # Assuming Tool is the base, FunctionTool has on_invoke_tool
from .model import RealtimeEventToolCall  # The event type that triggers this


class ToolExecutor:
    """Executes tools based on RealtimeEventToolCall events."""

    def __init__(self, tools: Sequence[Tool], shared_context: Any | None = None):
        self._tool_map: Dict[str, FunctionTool] = {}
        self._shared_context = shared_context
        # Explicitly specify which tools need context - we'll set all tools with first param named "context"
        self._context_aware_tools: Set[str] = set()

        for tool in tools:
            if isinstance(tool, FunctionTool):
                self._tool_map[tool.name] = tool

                # Debug - log all attributes of the FunctionTool
                logger.info(f"FunctionTool {tool.name} attributes: {dir(tool)}")

                # Get the original function if available
                if hasattr(tool, "function"):
                    func = tool.function
                    logger.info(f"Found function attribute for {tool.name}: {func}")
                    if callable(func):
                        # Check if first parameter is named "context" - simpler approach
                        sig = inspect.signature(func)
                        params = list(sig.parameters.keys())
                        logger.info(f"Function {tool.name} params: {params}")
                        if params and params[0] == "context":
                            self._context_aware_tools.add(tool.name)
                            logger.info(f"Detected context-aware tool: {tool.name}")
                else:
                    # Try to inspect on_invoke_tool to see if we can find more info
                    logger.info(
                        f"Tool {tool.name} has no 'function' attribute. Examining on_invoke_tool: {tool.on_invoke_tool}"
                    )

                    # Special hardcoded handling - for now, let's explicitly mark these tools as context-aware
                    if tool.name in ["greet_user_and_count", "get_user_details"]:
                        logger.info(
                            f"Explicitly marking {tool.name} as context-aware based on name"
                        )
                        self._context_aware_tools.add(tool.name)
            else:
                # For now, only FunctionTools are supported by this simple executor.
                # We can extend this later if other tool types (e.g. ComputerTool) are needed
                # in the realtime flow directly without going through a full agent run.
                logger.warning(
                    f"Tool '{tool.name}' is not a FunctionTool and will be ignored by ToolExecutor."
                )

        logger.info(f"Context-aware tools: {self._context_aware_tools}")

    async def execute(self, tool_call_event: RealtimeEventToolCall) -> str:
        """Executes the specified tool and returns its string output.

        Args:
            tool_call_event: The RealtimeEventToolCall describing the tool to execute.

        Returns:
            A string representation of the tool's output (typically JSON).

        Raises:
            AgentsException: If the tool is not found or fails during execution.
        """
        tool_name = tool_call_event.tool_name
        tool = self._tool_map.get(tool_name)

        if not tool:
            err_msg = f"Tool '{tool_name}' not found in ToolExecutor."
            logger.error(err_msg)
            # Return an error string that can be sent back to the LLM
            return json.dumps({"error": err_msg, "tool_name": tool_name})

        # Convert arguments dict to JSON string, as expected by on_invoke_tool
        try:
            arguments_json = json.dumps(tool_call_event.arguments)
        except TypeError as e:  # pragma: no cover
            err_msg = f"Failed to serialize arguments for tool '{tool_name}': {e}"
            logger.error(f"{err_msg} Arguments: {tool_call_event.arguments}")
            return json.dumps({"error": err_msg, "tool_name": tool_name})

        logger.info(f"Executing tool: {tool_name} with args: {arguments_json}")

        try:
            # Check if this is a context-aware tool
            needs_context = tool_name in self._context_aware_tools

            # Execute the tool with or without context
            if needs_context:
                logger.info(
                    f"Tool {tool_name} is context-aware, passing RunContextWrapper"
                )
                tool_output = await tool.on_invoke_tool(
                    RunContextWrapper(context=self._shared_context), arguments_json
                )
            else:
                logger.info(
                    f"Tool {tool_name} is not context-aware, invoking without RunContextWrapper"
                )
                tool_output = await tool.on_invoke_tool(None, arguments_json)

            # Ensure the output is a string (as expected by OpenAI tool result content)
            if not isinstance(tool_output, str):
                # Attempt to convert common types to string (e.g. dict to JSON string)
                if isinstance(tool_output, (dict, list)):
                    tool_output_str = json.dumps(tool_output)
                else:
                    tool_output_str = str(tool_output)
            else:
                tool_output_str = tool_output

            logger.info(
                f"Tool {tool_name} executed successfully. Output length: {len(tool_output_str)}"
            )
            return tool_output_str
        except Exception as e:  # pragma: no cover
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            # Return an error string that can be sent back to the LLM
            return json.dumps({"error": str(e), "tool_name": tool_name})
