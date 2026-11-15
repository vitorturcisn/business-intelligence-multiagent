"""A2A REST server entry point for an individual BI specialist."""

import argparse

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_rest_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette

from bi_multiagent.a2a.cards import (
    REST_PATH_PREFIX,
    build_agent_card,
)
from bi_multiagent.a2a.executor import SpecialistA2AExecutor
from bi_multiagent.a2a.registry import AGENT_SPECS, get_agent_spec


def create_agent_app(
    agent_name: str,
    base_url: str | None = None,
) -> Starlette:
    """Creates a Starlette application exposing one A2A specialist."""
    spec = get_agent_spec(agent_name)

    effective_base_url = base_url or f"http://127.0.0.1:{spec.port}"

    agent_card = build_agent_card(
        agent_name=agent_name,
        base_url=effective_base_url,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=SpecialistA2AExecutor(agent_name),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = [
        *create_agent_card_routes(agent_card),
        *create_rest_routes(
            request_handler,
            path_prefix=REST_PATH_PREFIX,
        ),
    ]

    return Starlette(routes=routes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Business Intelligence A2A specialist."
    )
    parser.add_argument(
        "--agent",
        choices=sorted(AGENT_SPECS),
        required=True,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--public-base-url",
        default=None,
        help="Public base URL advertised in the Agent Card.",
    )

    args = parser.parse_args()
    spec = get_agent_spec(args.agent)
    port = args.port or spec.port

    base_url = args.public_base_url or f"http://127.0.0.1:{port}"
    app = create_agent_app(args.agent, base_url=base_url)

    import uvicorn

    print(f"Starting A2A agent: {spec.display_name}")
    print(f"Agent Card: {base_url}/.well-known/agent-card.json")
    print(f"A2A REST: {base_url}{REST_PATH_PREFIX}")

    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
