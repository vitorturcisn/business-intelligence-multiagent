import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


class WorkerState(TypedDict, total=False):
    """Estado interno compartilhado pelos subgrafos especialistas."""

    question: str
    request_id: str
    task: dict[str, Any]
    schema: dict[str, Any]
    sql: str
    evidence: str
    sources: list[str]
    warnings: list[str]
    start_date: str
    end_date: str
    result: dict[str, Any]


class MultiAgentState(MessagesState, total=False):
    """Estado global do Supervisor e do Blackboard."""

    question: str
    request_id: str
    task: dict[str, Any]
    route: list[str]
    route_method: str
    route_rationale: str

    agent_results: Annotated[list[dict[str, Any]], operator.add]
    blackboard: Annotated[list[dict[str, Any]], operator.add]

    audit: dict[str, Any]
    answer: str

    # Human-in-the-Loop e autorização de ações
    approval_required: bool
    pending_action: dict[str, Any]
    approval: dict[str, Any]
    action_result: dict[str, Any]
