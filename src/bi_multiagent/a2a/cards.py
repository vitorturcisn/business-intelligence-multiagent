"""A2A 1.0 Agent Card definitions."""

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)

from bi_multiagent.a2a.registry import get_agent_spec

REST_PATH_PREFIX = "/a2a/rest"


def build_agent_card(agent_name: str, base_url: str) -> AgentCard:
    """Builds a protocol-compliant Protobuf AgentCard."""
    spec = get_agent_spec(agent_name)

    skill = AgentSkill(
        id=spec.skill_id,
        name=spec.skill_name,
        description=spec.skill_description,
        tags=list(spec.tags),
        examples=list(spec.examples),
        input_modes=["text/plain"],
        output_modes=["text/plain", "application/json"],
    )

    interface = AgentInterface(
        url=f"{base_url.rstrip('/')}{REST_PATH_PREFIX}",
        protocol_binding="HTTP+JSON",
        protocol_version="1.0",
    )

    return AgentCard(
        name=spec.display_name,
        description=spec.description,
        supported_interfaces=[interface],
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            extended_agent_card=False,
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        skills=[skill],
    )
