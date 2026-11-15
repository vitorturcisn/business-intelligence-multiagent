"""A2A client with Agent Card discovery and streamed task events."""

import argparse
import asyncio
import json

import httpx
from a2a.client import ClientConfig, create_client
from a2a.client.card_resolver import A2ACardResolver
from a2a.client.client import ClientCallContext
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest, TaskState
from google.protobuf.json_format import MessageToDict


async def send_request(agent_url: str, question: str) -> None:
    """Discovers an agent, sends a request and consumes its event stream."""
    timeout = httpx.Timeout(
        180.0,
        connect=10.0,
        read=180.0,
        write=30.0,
        pool=10.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as http_client:
        resolver = A2ACardResolver(
            httpx_client=http_client,
            base_url=agent_url.rstrip("/"),
        )
        card = await resolver.get_agent_card()

        print(f"Agent descoberto: {card.name}")
        print(f"Versão: {card.version}")

        client = await create_client(
            agent=card,
            client_config=ClientConfig(
                streaming=True,
                supported_protocol_bindings=["HTTP+JSON"],
                httpx_client=http_client,
            ),
        )

        request = SendMessageRequest(
            message=new_text_message(
                text=question,
                role=Role.ROLE_USER,
            )
        )

        try:
            async for event in client.send_message(
                request,
                context=ClientCallContext(timeout=180.0),
            ):
                if event.HasField("task"):
                    task = event.task
                    print(f"\\nTASK: {task.id}")
                    print(f"Context: {task.context_id}")

                elif event.HasField("status_update"):
                    state = event.status_update.status.state
                    try:
                        state_name = TaskState.Name(state)
                    except ValueError:
                        state_name = str(state)

                    print(f"\\nSTATUS: {state_name}")

                    status_message = event.status_update.status.message
                    if status_message:
                        from a2a.helpers import get_message_text

                        print(get_message_text(status_message))

                elif event.HasField("artifact_update"):
                    artifact = event.artifact_update.artifact
                    print(f"\\nARTIFACT: {artifact.name}")

                    for part in artifact.parts:
                        if part.text:
                            print(part.text)
                        elif part.HasField("data"):
                            payload = MessageToDict(part.data)
                            print(
                                json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                            )

                elif event.HasField("message"):
                    from a2a.helpers import get_message_text

                    print(get_message_text(event.message))

        finally:
            await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a request to a BI A2A specialist."
    )
    parser.add_argument(
        "--agent-url",
        required=True,
        help="Base URL of the A2A agent, e.g. http://127.0.0.1:9101",
    )
    parser.add_argument(
        "--question",
        required=True,
    )
    args = parser.parse_args()

    asyncio.run(
        send_request(
            agent_url=args.agent_url,
            question=args.question,
        )
    )


if __name__ == "__main__":
    main()
