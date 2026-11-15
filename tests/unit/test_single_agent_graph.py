from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver

from bi_multiagent.agents.single_agent_graph import (
    SYSTEM_PROMPT,
    build_single_agent_graph,
    retrieve_context,
)


class FakeModel:
    """Modelo determinístico para testar o grafo sem chamar Ollama."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_messages = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.seen_messages.append(list(messages))
        return self.responses.pop(0)


def test_retrieval_node_injects_real_sales_schema():
    result = retrieve_context(
        {"messages": [HumanMessage(content="Quais tabelas existem no banco?")]}
    )

    assert "sales" in result["evidence"]
    assert "table_names" in result["evidence"]


def test_retrieval_node_injects_documentary_evidence():
    result = retrieve_context(
        {
            "messages": [
                HumanMessage(content="Quais os principais indicadores de negócio?")
            ]
        }
    )

    assert "01_glossario_kpis.md" in result["evidence"]
    assert "Receita bruta" in result["evidence"]
    assert "CAC" not in result["evidence"]


def test_system_prompt_enforces_source_boundaries():
    assert "não invente" in SYSTEM_PROMPT.lower()
    assert "série 432" in SYSTEM_PROMPT
    assert "série 433" in SYSTEM_PROMPT
    assert "CAC" in SYSTEM_PROMPT


def test_graph_executes_react_tool_cycle():
    fake_model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_sales_schema",
                        "args": {},
                        "id": "schema-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="A tabela disponível é sales."),
        ]
    )

    graph = build_single_agent_graph(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    config = {
        "configurable": {
            "thread_id": "react-test",
        }
    }

    result = graph.invoke(
        {"messages": [HumanMessage(content="Quais tabelas existem?")]},
        config=config,
    )

    assert result["messages"][-1].content == ("A tabela disponível é sales.")
    assert any(isinstance(message, ToolMessage) for message in result["messages"])

    # Evidência do schema foi disponibilizada antes da chamada do LLM.
    first_system_message = fake_model.seen_messages[0][0]
    assert "table_names" in first_system_message.content


def test_checkpointer_preserves_history_by_thread():
    fake_model = FakeModel(
        [
            AIMessage(content="Primeira resposta."),
            AIMessage(content="Segunda resposta."),
        ]
    )

    graph = build_single_agent_graph(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    config = {
        "configurable": {
            "thread_id": "memory-test",
        }
    }

    graph.invoke(
        {"messages": [HumanMessage(content="Meu nome é Ana.")]},
        config=config,
    )

    graph.invoke(
        {"messages": [HumanMessage(content="Qual é meu nome?")]},
        config=config,
    )

    first_context_size = len(fake_model.seen_messages[0])
    second_context_size = len(fake_model.seen_messages[1])

    assert second_context_size > first_context_size
    assert any(
        getattr(message, "content", "") == "Meu nome é Ana."
        for message in fake_model.seen_messages[1]
    )
