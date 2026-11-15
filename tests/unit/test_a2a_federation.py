import asyncio

import pytest

from bi_multiagent.a2a.cards import build_agent_card
from bi_multiagent.a2a.executor import invoke_specialist
from bi_multiagent.a2a.registry import AGENT_SPECS, get_agent_spec
from bi_multiagent.schemas.contracts import AgentResult


@pytest.mark.parametrize(
    "agent_name",
    ["sales", "knowledge", "economic"],
)
def test_agent_card_has_a2a_10_http_json_interface(agent_name):
    card = build_agent_card(
        agent_name,
        f"http://127.0.0.1:{AGENT_SPECS[agent_name].port}",
    )

    assert card.name == AGENT_SPECS[agent_name].display_name
    assert card.version == "1.0.0"
    assert card.capabilities.streaming is True
    assert len(card.supported_interfaces) == 1
    assert card.supported_interfaces[0].protocol_binding == "HTTP+JSON"
    assert card.supported_interfaces[0].protocol_version == "1.0"
    assert card.skills[0].id == AGENT_SPECS[agent_name].skill_id


def test_registry_rejects_unknown_agent():
    with pytest.raises(ValueError, match="Unknown agent"):
        get_agent_spec("unknown")


class FakeGraph:
    def __init__(self, result):
        self.result = result
        self.received_state = None

    async def ainvoke(self, state):
        self.received_state = state
        return {"result": self.result}


def test_invoke_specialist_validates_structured_contract():
    expected = AgentResult(
        request_id="request-123",
        agent="sales",
        status="completed",
        summary="Net revenue calculated.",
        evidence='{"net_revenue": 43270.0}',
        sources=["data/sample/sales.db"],
        warnings=[],
    )

    graph = FakeGraph(expected.model_dump(mode="json"))

    result = asyncio.run(
        invoke_specialist(
            agent_name="sales",
            question="Calculate net revenue.",
            request_id="request-123",
            graph=graph,
        )
    )

    assert result == expected
    assert graph.received_state == {
        "question": "Calculate net revenue.",
        "request_id": "request-123",
    }


def test_invoke_specialist_rejects_agent_mismatch():
    result = AgentResult(
        request_id="request-123",
        agent="knowledge",
        status="completed",
        summary="Wrong agent.",
    )

    graph = FakeGraph(result.model_dump(mode="json"))

    with pytest.raises(ValueError, match="Expected result from"):
        asyncio.run(
            invoke_specialist(
                agent_name="sales",
                question="Calculate net revenue.",
                request_id="request-123",
                graph=graph,
            )
        )


def test_invoke_specialist_rejects_request_id_mismatch():
    result = AgentResult(
        request_id="other-request",
        agent="sales",
        status="completed",
        summary="Wrong request.",
    )

    graph = FakeGraph(result.model_dump(mode="json"))

    with pytest.raises(ValueError, match="request_id mismatch"):
        asyncio.run(
            invoke_specialist(
                agent_name="sales",
                question="Calculate net revenue.",
                request_id="request-123",
                graph=graph,
            )
        )
