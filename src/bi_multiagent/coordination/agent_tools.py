"""Expose compiled specialist subgraphs as structured LangChain tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from bi_multiagent.schemas.contracts import AgentName, AgentResult, AgentTask


class AgentToolInput(BaseModel):
    """Validated input contract accepted by every specialist tool."""

    task: AgentTask


@dataclass(slots=True)
class AgentTool:
    """Wrap one compiled specialist graph as a LangChain StructuredTool.

    The wrapper preserves the existing LangGraph fan-out architecture while
    giving the Supervisor a uniform, inspectable tool interface for each agent.
    """

    agent: AgentName
    objective: str
    graph: Any
    _tool: StructuredTool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tool = StructuredTool.from_function(
            func=self._run,
            name=f"{self.agent}_agent_tool",
            description=(
                f"Delegate a task to the {self.agent} specialist. "
                f"Domain objective: {self.objective} "
                "Input must be a validated AgentTask; "
                "output is a structured AgentResult."
            ),
            args_schema=AgentToolInput,
        )

    @property
    def tool(self) -> StructuredTool:
        """Return the LangChain tool for registration or inspection."""
        return self._tool

    def _run(self, task: AgentTask) -> dict[str, Any]:
        validated_task = AgentTask.model_validate(task)
        if validated_task.agent != self.agent:
            raise ValueError(
                f"Tool '{self.agent}' cannot execute task for agent "
                f"'{validated_task.agent}'."
            )

        output = self.graph.invoke(
            {
                "question": validated_task.question,
                "request_id": validated_task.request_id,
                "task": validated_task.model_dump(mode="json"),
            }
        )
        if not isinstance(output, dict) or "result" not in output:
            raise ValueError(
                f"Specialist '{self.agent}' returned an invalid graph payload."
            )

        result = AgentResult.model_validate(output["result"])
        if result.agent != self.agent:
            raise ValueError(
                f"Specialist '{self.agent}' returned result for '{result.agent}'."
            )
        if result.request_id != validated_task.request_id:
            raise ValueError(
                "Specialist result request_id does not match the submitted task."
            )

        return result.model_dump(mode="json")

    def invoke(self, task: AgentTask | dict[str, Any]) -> AgentResult:
        """Invoke the specialist through its LangChain tool interface."""
        validated_task = AgentTask.model_validate(task)
        payload = self._tool.invoke({"task": validated_task.model_dump(mode="json")})
        return AgentResult.model_validate(payload)
