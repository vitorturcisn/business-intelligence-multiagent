# Business Intelligence Multi-Agent System

Sistema de Business Intelligence construído com Python, LangChain e
LangGraph, evoluindo o agente individual para uma arquitetura multiagente.

## Arquitetura atual

- Agente individual com StateGraph e ciclo ReAct.
- Recuperação explícita de evidências.
- Supervisor hierárquico com roteamento estruturado.
- Fan-out paralelo com LangGraph Send.
- Sales Analyst como subgrafo.
- Knowledge Analyst como subgrafo.
- Economic Analyst como subgrafo.
- Contratos Pydantic AgentTask e AgentResult.
- Blackboard compartilhado com reducers.
- Audit Agent para validação básica de fontes.
- Memória por thread_id com InMemorySaver.

## Fontes de dados

- `data/sample/sales.db`: tabela `sales`.
- `data/knowledge_base/`: glossário, dicionário, regras e limitações.
- Banco Central do Brasil: SGS série 432 (meta Selic).

Os dados comerciais são sintéticos.

## Executar o agente individual

```powershell
python -m bi_multiagent.agents.cli --thread-id single-demo
```

## Executar o sistema multiagente

```powershell
python -m bi_multiagent.agents.multiagent_cli --thread-id multiagent-demo
```

## Testes

```powershell
python -m pytest -v
ruff check src tests
python -m pip check
```

## Limitações atuais

- Sales Analyst utiliza templates SQL determinísticos.
- Economic Analyst exige datas explícitas DD/MM/AAAA.
- Blackboard é mantido no estado do grafo.
- Auditoria atual verifica fontes e presença de evidências.
- Checkpointer local usa InMemorySaver.

## Próximas fases

- Consenso ponderado e auditoria semântica.
- Comunicação A2A.
- Human-in-the-loop.
- LangSmith e avaliações.
- Supabase PGVector e implantação.