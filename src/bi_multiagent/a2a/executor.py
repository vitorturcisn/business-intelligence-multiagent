"""A2A AgentExecutor adapter for the existing LangGraph specialists."""

import logging
from typing import Any
from uuid import uuid4

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Role, TaskState
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Value

from bi_multiagent.a2a.registry import get_agent_spec
from bi_multiagent.schemas.contracts import AgentResult

logger = logging.getLogger(__name__)


def build_specialist_graph(agent_name: str):
    """Builds the existing specialist subgraph without changing its logic."""
    get_agent_spec(agent_name)

    from bi_multiagent.agents.specialized_agents import (
        build_economic_analyst_graph,
        build_knowledge_analyst_graph,
        build_sales_analyst_graph,
    )

    factories = {
        "sales": build_sales_analyst_graph,
        "knowledge": build_knowledge_analyst_graph,
        "economic": build_economic_analyst_graph,
    }

    return factories[agent_name]()


async def invoke_specialist(
    agent_name: str,
    question: str,
    request_id: str,
    graph: Any | None = None,
) -> AgentResult:
    """Invokes a specialist subgraph and validates its structured result."""
    active_graph = graph if graph is not None else build_specialist_graph(agent_name)

    state = await active_graph.ainvoke(
        {
            "question": question,
            "request_id": request_id,
        }
    )

    raw_result = state.get("result")

    if raw_result is None:
        raise RuntimeError(f"Specialist '{agent_name}' returned no structured result.")

    result = AgentResult.model_validate(raw_result)

    if result.agent != agent_name:
        raise ValueError(
            f"Expected result from '{agent_name}', received '{result.agent}'."
        )

    if result.request_id != request_id:
        raise ValueError("Specialist result request_id mismatch.")

    return result


class SpecialistA2AExecutor(AgentExecutor):
    """Adapts one LangGraph specialist to the A2A task lifecycle."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        get_agent_spec(agent_name)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Runs the specialist and streams task status and result artifacts."""
        message = context.message

        if message is None:
            raise ValueError("A2A request does not contain a message.")

        task = context.current_task

        if task is None:
            task = new_task_from_user_message(message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )

        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message(
                f"{self.agent_name} agent started processing.",
                role=Role.ROLE_AGENT,
            ),
        )

        try:
            question = get_message_text(message).strip()

            if not question:
                raise ValueError("A2A request contains no text input.")

            request_id = uuid4().hex
            result = await invoke_specialist(
                agent_name=self.agent_name,
                question=question,
                request_id=request_id,
            )

            payload = result.model_dump(mode="json")

            structured_part = Part(
                data=ParseDict(payload, Value()),
            )

            await updater.add_artifact(
                parts=[
                    new_text_part(
                        text=result.summary,
                        media_type="text/plain",
                    ),
                    structured_part,
                ],
                name=f"{self.agent_name}_result",
                metadata={
                    "request_id": result.request_id,
                    "agent": result.agent,
                },
            )

            final_message = new_text_message(
                f"{self.agent_name} agent finished processing.",
                role=Role.ROLE_AGENT,
            )

            if result.status == "failed":
                final_state = TaskState.TASK_STATE_FAILED
            elif result.status == "needs_input":
                final_state = TaskState.TASK_STATE_INPUT_REQUIRED
            else:
                final_state = TaskState.TASK_STATE_COMPLETED

            await updater.update_status(
                state=final_state,
                message=final_message,
            )

        except Exception as exc:
            logger.exception(
                "A2A specialist execution failed: %s",
                self.agent_name,
            )

            await updater.update_status(
                state=TaskState.TASK_STATE_FAILED,
                message=new_text_message(
                    f"Specialist execution failed ({type(exc).__name__}).",
                    role=Role.ROLE_AGENT,
                ),
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Publishes cancellation for the current A2A task."""
        task = context.current_task

        if task is None:
            return

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )

        await updater.cancel(
            new_text_message(
                f"{self.agent_name} task was canceled.",
                role=Role.ROLE_AGENT,
            )
        )
