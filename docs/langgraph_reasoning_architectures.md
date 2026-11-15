# Arquiteturas de raciocínio — Business Intelligence Multi-Agent

## 1. Classificação do agente original

O agente individual implementado em `single_agent_graph.py` é classificado
principalmente como um Router com preparação sequencial de evidências.

A preparação inicial é uma Chain:

1. Recebe a mensagem do usuário.
2. Recupera schema ou documentação quando necessário.
3. Injeta as evidências no contexto.

Depois, o modelo pode selecionar ferramentas. A função `tools_condition`
direciona a execução para `ToolNode` ou encerra a resposta. Quando uma
ferramenta é executada, o grafo retorna ao agente.

Portanto, a arquitetura completa é híbrida: Chain para preparação e Router
para seleção de ferramentas, em um ciclo ReAct.

## 2. ReAct existente

O grafo ReAct utiliza:

- `retrieve_context` para obter evidências.
- `agent` para raciocínio e seleção de ferramentas.
- `ToolNode` para executar as ferramentas.
- `tools_condition` para decidir entre executar uma ferramenta ou finalizar.
- Checkpointer para persistir o histórico por thread.
- `trim_messages` para limitar o histórico enviado ao modelo.

O agente decide dinamicamente qual ferramenta usar em cada ciclo.

## 3. Plan-and-Execute implementado

O grafo alternativo está em `plan_execute_graph.py`.

Sua sequência é:

1. `retrieve_context`: recupera schema ou documentação.
2. `planner`: gera uma sequência estruturada de etapas.
3. `execute_step`: executa uma etapa usando uma ferramenta autorizada.
4. `replan`: avalia as observações e decide se faltam evidências.
5. `finalize`: consolida as evidências em uma resposta.

O planejamento é limitado a cinco etapas por ciclo e dois ciclos adicionais
de replanejamento por execução.

O plano utiliza nomes e argumentos de ferramentas previamente permitidas.
O modelo não pode fornecer SQL executável nem criar ferramentas.

## 4. Comparação conceitual

| Critério | ReAct | Plan-and-Execute |
|---|---|---|
| Estratégia | Alterna raciocínio e ação | Planeja antes de executar |
| Seleção de ferramentas | A cada ciclo | Por etapas do plano |
| Visibilidade | Histórico de tool calls | Plano, etapas e observações |
| Mudança de estratégia | Durante o ciclo ReAct | No nó de replanejamento |
| Controle | Ferramentas permitidas | Ferramentas permitidas e limite de etapas |
| Custo potencial | Mais ciclos de raciocínio | Planejamento inicial e eventual replanejamento |

## 5. Adequação ao domínio de BI

ReAct é adequado para perguntas exploratórias e consultas em que a próxima
ferramenta depende diretamente da evidência recém-obtida.

Plan-and-Execute é adequado para solicitações compostas que podem ser
decompostas em etapas verificáveis, como recuperar uma definição, consultar
um indicador e depois consolidar as evidências.

A arquitetura não assume que Plan-and-Execute é sempre superior. A escolha
depende do tipo de pergunta e deve ser avaliada por qualidade, latência,
número de chamadas e completude das evidências.

## 6. Limitações e avaliação pendente

A implementação atual usa os mesmos especialistas e ferramentas do projeto.
O executor é sequencial e não executa SQL gerado pelo modelo.

A comparação empírica com ReAct ainda deve ser registrada em um dataset
LangSmith, com perguntas equivalentes e métricas de resposta, passos e
trajetória.


## Discount fraction diagnosis

LangSmith traces exposed SQL that divided discount_pct by 100.
The SQLite database stores discount_pct as a decimal fraction:
0.10 represents 10% and 0.05 represents 5%.

The validated net revenue formula is:
quantity * unit_price * (1 - discount_pct).

The model may supply SQL only as the argument of query_sales_database.
SQLite authorization enforces read-only execution.
