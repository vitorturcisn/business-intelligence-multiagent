"""LangGraph Agent Server entrypoint for the BI Multi-Agent Supervisor.

Agent Server supplies its own durable checkpointer in deployed environments.
Do not attach the local InMemorySaver or a second PostgresSaver here.
"""

from bi_multiagent.coordination.supervisor_graph import build_multiagent_graph

# Agent Server injects persistence for threads/checkpoints at runtime.
graph = build_multiagent_graph(use_in_memory_saver=False)
