# ADR 001 - Reconstrução do agente individual com StateGraph

## Contexto

O agente anterior concentrava roteamento, ferramentas, memória e resposta
em um runtime integrado. Para evoluir para uma arquitetura multiagente,
o fluxo individual precisa ter estado e transições explícitas.

## Decisão

Reconstruir o agente como um LangGraph StateGraph com:

- Estado conversacional baseado em MessagesState.
- Nó `agent` para chamada do modelo.
- Nó `tools` baseado em ToolNode.
- Roteamento condicional via tools_condition.
- Ciclo ReAct `agent -> tools -> agent`.
- Checkpointer InMemorySaver no ambiente local.
- Identificação de conversas por thread_id.
- Histórico limitado antes da chamada ao modelo.

## Ferramentas

- `get_sales_schema`: leitura do schema do SQLite.
- `query_sales_database`: SQL SELECT somente leitura.
- `search_business_knowledge`: consulta aos documentos Markdown.
- `get_bcb_series`: consulta às séries temporais do Banco Central.

## Segurança

A consulta SQL permite apenas SELECT ou WITH ... SELECT.
O SQLite é configurado com query_only e um authorizer que bloqueia
operações não classificadas como leitura.

## Próxima evolução

Na fase de produção, substituir o checkpointer em memória por PostgreSQL.
Na arquitetura multiagente, os agentes especializados serão subgrafos
coordenados por um supervisor.