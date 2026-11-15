from langgraph.graph import MessagesState


class SingleAgentState(MessagesState):
    """Estado conversacional e evidências recuperadas para a solicitação."""

    evidence: str
