# ADR 003 - Supervisor hierárquico com subgrafos especializados

## Contexto

O agente individual já possui StateGraph, ciclo ReAct, ferramentas,
memória por thread_id e recuperação de evidências.

Perguntas que combinam análise comercial, documentação e indicadores
econômicos exigem separação de responsabilidades e cooperação.

## Decisão

Adotar um padrão Supervisor hierárquico com fan-out paralelo via Send.

O Supervisor utiliza saída estruturada Pydantic para escolher:
- Sales Analyst
- Knowledge Analyst
- Economic Analyst

Cada especialista é implementado como um subgrafo LangGraph.

## Contratos

AgentTask padroniza a solicitação enviada pelo Supervisor.

AgentResult padroniza o retorno:
- request_id
- agent
- status
- summary
- evidence
- sources
- warnings

SupervisorDecision define a lista de agentes e a justificativa do roteamento.

## Blackboard

O estado global mantém:
- decisão do Supervisor;
- resultados dos especialistas;
- evidências e respectivas fontes;
- relatório de auditoria.

Reducers agregam resultados de agentes executados em paralelo.
Cada artefato possui request_id para separar solicitações na mesma thread.

## Audit

O Audit Agent verifica presença de evidências e compatibilidade básica
das fontes declaradas:
- sales -> SQLite sales.db;
- knowledge -> documentos Markdown;
- economic -> BCB SGS série 432.

A auditoria não representa verificação semântica completa ou NLI.
Essa evolução será documentada em uma próxima decisão.

## Contingência

O Supervisor usa structured output com ChatOllama.
Se o roteamento estruturado falhar, é acionado fallback determinístico
por palavras-chave. O método utilizado fica registrado.

## Limitações atuais

- O Sales Analyst usa templates SQL determinísticos para os indicadores
  documentados; não gera SQL arbitrário via LLM.
- O Economic Analyst exige duas datas explícitas DD/MM/AAAA.
- O Blackboard opera no estado do grafo e não em banco externo.
- O Audit verifica fontes e evidências, mas não executa consenso ponderado.
- O checkpointer local é InMemorySaver.

## Próximas evoluções

- Consenso ponderado e auditoria semântica.
- Comunicação A2A com Agent Cards e tarefas assíncronas.
- Human-in-the-loop com interrupt e resume.
- Persistência PostgreSQL e Supabase.
- LangSmith para tracing e avaliação de trajetórias.