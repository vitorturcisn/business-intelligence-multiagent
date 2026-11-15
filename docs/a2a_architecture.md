# A2A Federation — Business Intelligence Multi-Agent

## Protocol

The federation uses the official A2A Python SDK and A2A Protocol 1.0.

Transport: HTTP+JSON/REST with streamed task events.

Agent Cards are published at:

`/.well-known/agent-card.json`

REST endpoints are mounted under:

`/a2a/rest`

## Specialists

| Agent | Default port | Responsibility |
|---|---:|---|
| Sales | 9101 | Sales indicators, read-only SQL and validated Text-to-Query |
| Knowledge | 9102 | Business documentation, FAISS retrieval and rules |
| Economic | 9103 | Historical Selic from BCB SGS 432 |

Each A2A executor wraps the existing LangGraph specialist subgraph.
The business logic remains in `specialized_agents.py`.

## Task lifecycle

The executor publishes the initial A2A Task, then a WORKING status,
a structured artifact and a terminal status.

The artifact contains:
- A text summary.
- A JSON `AgentResult` with request ID, agent, status, evidence,
  sources and warnings.

The task store is currently in-memory for this local prototype.
The main conversational graph continues to use PostgreSQL checkpointing.

## Streaming

The client uses the official A2A client and consumes `StreamResponse`
events as they arrive:
- Task
- TaskStatusUpdateEvent
- TaskArtifactUpdateEvent

## Local execution

Start each agent in a separate terminal:

```powershell
python -m bi_multiagent.a2a.server --agent sales
python -m bi_multiagent.a2a.server --agent knowledge
python -m bi_multiagent.a2a.server --agent economic
```

Send a request:

```powershell
python -m bi_multiagent.a2a.client --agent-url http://127.0.0.1:9101 --question "Calculate total net revenue."
```

## Current limitations

- The A2A task store is in-memory; task state is not durable across a server restart.
- The local demonstration does not configure authentication or TLS.
- The A2A client is available as a separate integration layer; the main
  Supervisor has not yet been switched from local subgraph invocation.
- A2A event streaming reports task progress and final artifacts. It does not
  yet forward token-level LLM streaming from the specialist subgraphs.
