# Business Intelligence Multi-Agent System

Sistema de Business Intelligence desenvolvido em Python, LangChain e LangGraph. O projeto evolui um agente individual para uma arquitetura hierárquica multiagente capaz de consultar dados comerciais sintéticos, recuperar regras de negócio e consultar a série histórica da meta Selic do Banco Central do Brasil.

> **Estado do projeto:** este README descreve o código presente no repositório e distingue funcionalidades implementadas de itens ainda em evolução. Não considere deployment gerenciado, Supabase PGVector, consenso ponderado ou auditoria completa de fidelidade às instruções como concluídos nesta versão.

## 1. Domínio e fontes de dados

O sistema atende perguntas de Business Intelligence sobre vendas, indicadores, fórmulas de negócio e meta Selic.

| Fonte | Conteúdo | Uso |
|---|---|---|
| `data/sample/sales.db` | Banco SQLite sintético, tabela `sales` | Consultas e agregações comerciais somente leitura |
| `data/knowledge_base/` | Glossário, dicionário de dados, regras e limitações | Recuperação de evidências documentais |
| Banco Central do Brasil — SGS 432 | Série histórica da meta Selic | Consultas econômicas com intervalo explícito |

Os dados comerciais são sintéticos e são gerados localmente por um seed determinístico. O arquivo `.db` não precisa ser versionado no Git.

### Regra importante sobre descontos

`discount_pct` é armazenado como fração decimal: `0.10` representa 10% e `0.05` representa 5%. A receita líquida por item é calculada por:

```text
quantity * unit_price * (1 - discount_pct)
```

Não divida `discount_pct` por 100 novamente.

## 2. Arquitetura

O pacote principal está em `src/bi_multiagent/` e contém:

- **Agente individual:** `StateGraph`, recuperação prévia de contexto, ciclo ReAct com `ToolNode` e `tools_condition`, histórico limitado com `trim_messages` e checkpoint configurável. O padrão local utiliza `InMemorySaver`.
- **Supervisor:** roteamento estruturado para os especialistas, com política determinística de proteção e fallback por palavras-chave.
- **Sales Analyst:** subgrafo para métricas comerciais, templates SQL determinísticos e integração com o Text-to-Query validado.
- **Knowledge Analyst:** subgrafo de recuperação da base documental, combinando o índice FAISS existente com recuperação documental complementar.
- **Economic Analyst:** subgrafo para consulta à série SGS 432 do Banco Central.
- **Coordenação:** fan-out paralelo por `Send`, contratos Pydantic `AgentTask`, `AgentResult` e `SupervisorDecision`, e estado compartilhado no padrão Blackboard.
- **Audit Agent:** verifica presença de evidências e compatibilidade básica das fontes declaradas. Não equivale, ainda, a consenso ponderado nem a uma auditoria semântica completa.
- **Human-in-the-loop:** `interrupt()` pausa uma solicitação de gravação de relatório; `Command(resume=...)` recebe aprovação/rejeição. O executor só grava o arquivo local após autorização.
- **Concorrência:** o CLI multiagente utiliza advisory lock PostgreSQL por `thread_id`; uma segunda execução concorrente da mesma thread é rejeitada.
- **A2A:** servidores especializados publicam Agent Cards e recebem tarefas pelo SDK A2A. Esta integração é separada: o Supervisor principal ainda invoca subgrafos locais diretamente.
- **Raciocínio alternativo:** `plan_execute_graph.py` implementa Plan-and-Execute para comparação com ReAct.

A documentação complementar está em `docs/`, incluindo decisões arquiteturais, protocolo A2A, governança humana e comparação de arquiteturas de raciocínio.

## 3. Requisitos

- Python **3.12** (o `pyproject.toml` restringe a versão a `>=3.12,<3.13`).
- Git.
- Ollama instalado e em execução para os fluxos que utilizam o modelo local.
- PostgreSQL acessível para o CLI multiagente, persistência e lock por thread.
- Chave LangSmith para publicar datasets/experimentos e habilitar tracing remoto.
- Acesso à internet para instalar dependências e consultar a API pública do Banco Central.

Modelo local padrão: `qwen3:1.7b` via Ollama.

## 4. Instalação local — Windows PowerShell

Clone o repositório e entre na pasta:

```powershell
git clone https://github.com/vitorturcisn/business-intelligence-multiagent.git
Set-Location business-intelligence-multiagent
```

Crie e ative o ambiente virtual:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Instale o projeto com os extras de desenvolvimento, recuperação e NLI:

```powershell
python -m pip install -e ".[dev,retrieval,nli]"
```

Crie o arquivo local de configuração e edite os valores necessários:

```powershell
Copy-Item .env.example .env
notepad .env
```

Nunca coloque credenciais reais no `.env.example` nem faça commit do `.env`.

Instale/inicie o modelo local:

```powershell
ollama pull qwen3:1.7b
```

Confirme que o serviço Ollama está disponível em `http://localhost:11434`.

## 5. Gerar o banco SQLite sintético

O gerador determinístico está em `src/bi_multiagent/data/seed.py`. Ele cria 500 identificadores de pedido e seus itens, com valores e datas reprodutíveis para a mesma seed.

Em um clone novo, gere o banco com:

```powershell
python -m bi_multiagent.data.seed
```

O destino padrão é `data/sample/sales.db`. O gerador **não sobrescreve** um banco existente. Para recriá-lo intencionalmente, faça backup e use:

```powershell
Copy-Item data/sample/sales.db "$HOME\Downloads\sales-backup.db"
python -m bi_multiagent.data.seed --force
```

A fixture `tests/conftest.py` também cria o banco automaticamente quando ele está ausente durante a execução dos testes. Ela não substitui um arquivo existente com schema válido.

## 6. Configuração PostgreSQL e persistência

O runtime do CLI multiagente utiliza `POSTGRES_CHECKPOINTER_URI` para o `PostgresSaver` e para o advisory lock. Configure essa variável no `.env` com uma URI PostgreSQL válida. Exemplo ilustrativo — substitua host, usuário e senha pelos dados do seu serviço:

```dotenv
POSTGRES_CHECKPOINTER_URI=postgresql://USER:PASSWORD@HOST:5432/postgres?sslmode=require
```

Inicialize as tabelas de checkpoint:

```powershell
python -m bi_multiagent.persistence.setup
```

O teste `scripts/verify_postgres_persistence.py` verifica a recuperação de uma conversa depois de fechar e reconstruir o pool/conexão. Execute-o somente quando o PostgreSQL estiver configurado e acessível:

```powershell
python scripts/verify_postgres_persistence.py
```

O agente individual usa `InMemorySaver` por padrão. O CLI multiagente abre o checkpointer PostgreSQL e exige a variável acima.

## 7. Executar o sistema

### Agente individual (ReAct)

```powershell
python -m bi_multiagent.agents.cli --thread-id single-demo
```

### Sistema multiagente com Supervisor

Com PostgreSQL configurado e tabelas inicializadas:

```powershell
python -m bi_multiagent.agents.multiagent_cli --thread-id multiagent-demo
```

Para exercitar o checkpoint humano, faça uma solicitação explícita para salvar o relatório em arquivo. O sistema deve pausar e pedir aprovação ou rejeição antes de gravar em `outputs/approved_reports/`.

### Plan-and-Execute

O grafo alternativo está em `bi_multiagent.agents.plan_execute_graph`. O script `scripts/evaluate_reasoning_architectures.py` compara ReAct e Plan-and-Execute no LangSmith. Consulte a seção de avaliação antes de executá-lo, pois ele publica/usa um dataset remoto e requer credenciais.

## 8. Integração A2A

A integração A2A pode executar cada especialista em um servidor separado. Abra três terminais, com o mesmo ambiente virtual ativo:

```powershell
python -m bi_multiagent.a2a.server --agent sales
```

```powershell
python -m bi_multiagent.a2a.server --agent knowledge
```

```powershell
python -m bi_multiagent.a2a.server --agent economic
```

Em outro terminal, envie uma solicitação para um especialista:

```powershell
python -m bi_multiagent.a2a.client --agent-url http://127.0.0.1:9101 --question "Calculate total net revenue."
```

Os Agent Cards são publicados em `/.well-known/agent-card.json`; os endpoints REST ficam sob `/a2a/rest`. O cliente consome eventos de tarefa e artefatos via streaming A2A.

**Limitações atuais do A2A:** o armazenamento de tarefas é em memória; a demonstração local não configura autenticação nem TLS; o Supervisor principal ainda não usa o cliente A2A para delegar; e o streaming comunica eventos de tarefa/artefatos, não tokens do modelo em tempo real.

## 9. LangSmith e avaliação

Configure no `.env`:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY= SUA_CHAVE_LANGSMITH
LANGSMITH_PROJECT=business-intelligence-multiagent
```

O script `scripts/evaluate_reasoning_architectures.py` cria/usa o dataset `bi-multiagent-reasoning-benchmark-v1` e avalia as arquiteturas ReAct e Plan-and-Execute com três dimensões:

- `response_fidelity`: fidelidade da resposta a termos e valores de referência;
- `step_execution`: execução das ferramentas/passos e limites de execução;
- `trajectory_completeness`: completude da trajetória e sequência de nós.

Execute com:

```powershell
python scripts/evaluate_reasoning_architectures.py
```

O dataset atual do script é um benchmark de raciocínio individual. Ele **não substitui** uma avaliação LangSmith específica do Supervisor multiagente. A avaliação dedicada da coordenação, com agentes esperados por cenário, fontes, transições e auditoria, permanece como trabalho a completar.

As evidências históricas do agente anterior estão preservadas em `docs/baseline_previous_agent/` e `evals/baseline_previous_agent/`. Elas devem ser identificadas como baseline anterior, não como resultados automaticamente reproduzidos pelo sistema multiagente atual.

## 10. Testes e qualidade

Execute os testes:

```powershell
python -m pytest -q
```

Verifique dependências:

```powershell
python -m pip check
```

Ruff nos arquivos novos do seed/fixture:

```powershell
python -m ruff check src/bi_multiagent/data/seed.py tests/conftest.py tests/unit/test_seed_sales_db.py
```

A validação local registrada em 15/11/2026 executou **96 testes aprovados** com o banco SQLite preexistente, incluindo os dois testes novos de seed, com dois avisos de depreciação. Uma validação adicional em clone sem `sales.db` identificou dois testes antigos que comparavam os resultados com totais fixos do banco legado, incompatíveis com o banco sintético gerado pelo seed. Esses testes foram atualizados para calcular a referência diretamente das linhas do banco; a execução limpa precisa ser repetida após a substituição do arquivo. Ruff passou nos três arquivos novos e `pip check` não encontrou requisitos quebrados. Na mesma execução, `ruff check src tests` apontou 72 ocorrências preexistentes em diferentes módulos do projeto legado `business_agent` e outros arquivos; portanto, não se declara lint global limpo nesta versão.

Para conferir o estado completo, execute também:

```powershell
python -m compileall -q src tests
```

## 11. Estrutura do repositório

```text
.
├── data/
│   └── knowledge_base/
├── docs/
│   ├── a2a_architecture.md
│   ├── human_in_the_loop.md
│   ├── langgraph_reasoning_architectures.md
│   ├── decisions/
│   └── baseline_previous_agent/
├── evals/
│   └── baseline_previous_agent/
├── scripts/
├── src/
│   ├── bi_multiagent/
│   │   ├── a2a/
│   │   ├── agents/
│   │   ├── coordination/
│   │   ├── persistence/
│   │   ├── schemas/
│   │   └── tools/
│   └── business_agent/
├── tests/
│   └── unit/
├── .env.example
├── pyproject.toml
└── requirements.txt
```

`business_agent/` preserva componentes do agente anterior que ainda são reutilizados, incluindo ferramentas, memória, recuperação FAISS e verificação. `bi_multiagent/` contém a arquitetura LangGraph individual e multiagente.

## 12. Limitações e próximos passos

A versão documentada apresenta as seguintes limitações ou itens em desenvolvimento:

1. **Deployment gerenciado:** ainda não há configuração de LangGraph Platform/Agent Server publicada como deployment reproduzível.
2. **Supabase PGVector:** a recuperação atual usa FAISS e recuperação documental complementar. A configuração e o vector store Supabase/PGVector de produção ainda precisam ser implementados.
3. **Consenso ponderado:** o Audit Agent verifica evidências e fontes, mas ainda não implementa consenso ponderado entre agentes.
4. **Instruction Fidelity Auditing:** a auditoria atual é básica e não avalia integralmente a aderência da resposta às instruções originais. Também não há Watchdog com retry adaptativo implementado.
5. **AgentTool:** os subgrafos especialistas são chamados diretamente pelo Supervisor; o encapsulamento formal como `AgentTool` ainda precisa ser concluído.
6. **Avaliação multiagente:** falta dataset e avaliadores LangSmith específicos do Supervisor, incluindo resultados por resposta, passo e trajetória da coordenação.
7. **Tracing de incidente:** é necessário vincular o diagnóstico histórico do erro de `discount_pct` a um trace verificável da execução que apresentou a falha e à execução corrigida.
8. **A2A em produção:** falta conectar a camada A2A ao fluxo de delegação do Supervisor e adicionar autenticação, TLS e persistência durável de tarefas.

## 13. Segurança e uso responsável

- Não versionar `.env`, chaves, senhas, bancos locais, índices gerados ou saídas de execução.
- O SQL comercial deve permanecer somente leitura e limitado às consultas autorizadas pela implementação.
- As consultas econômicas devem usar a série documentada e informar períodos/fontes.
- A gravação de relatórios depende de autorização humana explícita.
- A demonstração é um protótipo acadêmico; não deve ser tratada como um serviço corporativo de produção sem autenticação, autorização, monitoramento, gestão de segredos e controles operacionais adicionais.

## 14. Licença e contexto acadêmico

Projeto desenvolvido para a disciplina **Agentes Cognitivos com LangChain [26E4_2]**. O trabalho utiliza ferramentas de IA como apoio ao desenvolvimento e à documentação; as decisões, validações e evidências apresentadas devem ser verificadas pelo autor.

## Banco SQLite alternativo para benchmark

A variável opcional `BI_MULTIAGENT_DATABASE_PATH` seleciona o arquivo SQLite usado pelos módulos de vendas e pelo benchmark. Quando não está definida ou está vazia, o sistema mantém o caminho padrão `data/sample/sales.db`.

Para validar uma base sintética separada sem substituir o banco local existente:

```powershell
python -m bi_multiagent.data.seed --output .\data\sample\sales_benchmark_v2.db
$env:BI_MULTIAGENT_DATABASE_PATH = (Resolve-Path .\data\sample\sales_benchmark_v2.db).Path
python .\scripts\evaluate_reasoning_architectures.py --dry-run
```

Para voltar ao banco padrão na sessão atual do PowerShell:

```powershell
Remove-Item Env:BI_MULTIAGENT_DATABASE_PATH -ErrorAction SilentlyContinue
```
