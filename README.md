# Business Intelligence Multi-Agent System

Sistema de Business Intelligence desenvolvido em Python, LangChain e LangGraph. A aplicação usa um Supervisor para coordenar agentes especializados em vendas, conhecimento de negócio e indicadores econômicos.

> **Estado desta versão:** a arquitetura multiagente, AgentTool, Blackboard, consenso ponderado de evidências, auditoria de fidelidade, HITL, persistência PostgreSQL, recuperação Supabase PGVector e avaliação LangSmith estão implementados. A configuração do LangGraph Agent Server está versionada; isso não significa que exista um deployment cloud publicado.

## 1. Domínio e fontes

| Fonte | Conteúdo | Uso |
|---|---|---|
| SQLite `data/sample/sales.db` | Dados sintéticos da tabela `sales` | Indicadores comerciais, consultas SQL somente leitura e Text-to-Query validado |
| `data/knowledge_base/` | Glossário, dicionário, regras e limitações | Recuperação de fórmulas e regras de negócio |
| Banco Central do Brasil — SGS 432 | Série histórica da meta Selic | Consultas econômicas com intervalo explícito |
| Supabase PostgreSQL + pgvector | Embeddings dos documentos Markdown | Recuperação semântica opcional para o Knowledge Analyst |

Os dados de vendas são sintéticos e reproduzíveis pelo seed local. Não representam transações de uma empresa real.

### Regra de desconto

`discount_pct` é armazenado como fração decimal: `0.10` representa 10% e `0.05` representa 5%.

A receita líquida por item é:

```text
quantity * unit_price * (1 - discount_pct)
```

Não divida `discount_pct` por 100 novamente.

## 2. Arquitetura

O pacote principal está em `src/bi_multiagent/`.

- **Supervisor:** recebe a solicitação, produz uma decisão estruturada e seleciona os especialistas necessários. Há uma política determinística de proteção de roteamento e um fallback por palavras-chave.
- **Sales Analyst:** consulta schema e indicadores comerciais. Usa templates SQL de somente leitura e Text-to-Query com plano validado por catálogo fechado.
- **Knowledge Analyst:** combina recuperação semântica (FAISS ou Supabase PGVector) com recuperação documental complementar.
- **Economic Analyst:** consulta a série SGS 432 do Banco Central quando há um período explícito.
- **AgentTool:** encapsula cada subgrafo especialista como uma ferramenta LangChain estruturada, com contratos `AgentTask` e `AgentResult`.
- **Blackboard:** compartilha resultados e artefatos da execução entre Supervisor, especialistas, auditoria e síntese.
- **Consenso ponderado:** calcula suporte de evidências com pesos por domínio, completude, fontes e alertas. É uma pontuação determinística de suporte, não probabilidade calibrada nem votação semântica de verdade.
- **Instruction Fidelity Audit:** verifica aderência entre intenção, agentes selecionados, resultados, fontes, síntese e política HITL. É uma auditoria determinística, não uma garantia de factualidade.
- **Human-in-the-loop:** pedidos explícitos para salvar/exportar relatório pausam com `interrupt()`. A escrita só ocorre após aprovação humana; rejeições não criam arquivo.
- **Persistência:** o CLI multiagente usa `PostgresSaver` e advisory lock PostgreSQL por `thread_id`.
- **A2A:** servidores e cliente A2A estão disponíveis para executar especialistas separadamente. A camada A2A é independente: o Supervisor principal ainda delega aos subgrafos locais.
- **Raciocínio alternativo:** `plan_execute_graph.py` mantém a arquitetura Plan-and-Execute para comparação com ReAct.

## 3. Requisitos e instalação

- Python `>=3.12,<3.13`.
- Git.
- Ollama em execução para fluxos com modelo local.
- PostgreSQL para persistência do CLI multiagente.
- Chave LangSmith para tracing e avaliações remotas.
- URI de banco Supabase para ativar PGVector.

No Windows PowerShell:

```powershell
git clone https://github.com/vitorturcisn/business-intelligence-multiagent.git
Set-Location business-intelligence-multiagent

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,retrieval]"
Copy-Item .env.example .env
notepad .env
ollama pull qwen3:1.7b
```

O extra `nli` é opcional e instala dependências adicionais de Transformers/PyTorch:

```powershell
python -m pip install -e ".[nli]"
```

Nunca versione `.env`, tokens, senhas ou URIs com credenciais.

## 4. Configuração

Edite somente o `.env` local.

### Ollama

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
```

### LangSmith

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=business-intelligence-supervisor
LANGSMITH_WORKSPACE_ID=
```

O workspace ID é opcional, conforme o tipo de chave da organização.

### PostgreSQL do checkpointer

```dotenv
POSTGRES_CHECKPOINTER_URI=postgresql://USER:PASSWORD@HOST:5432/postgres?sslmode=require
```

Inicialize as tabelas:

```powershell
python -m bi_multiagent.persistence.setup
```

### Supabase PGVector

```dotenv
SUPABASE_DB_URI=postgresql://USER:PASSWORD@HOST:5432/postgres?sslmode=require
SUPABASE_SSLMODE=require
SUPABASE_VECTOR_SCHEMA=public
SUPABASE_VECTOR_TABLE=business_knowledge_vectors
SUPABASE_VECTOR_SIZE=384
KNOWLEDGE_VECTOR_BACKEND=faiss
```

`SUPABASE_DB_URI` é uma URI PostgreSQL, não a chave REST `SUPABASE_KEY`.

## 5. Banco SQLite sintético

O seed determinístico está em `src/bi_multiagent/data/seed.py`. Em clone novo:

```powershell
python -m bi_multiagent.data.seed
```

O destino padrão é `data/sample/sales.db`. O seed não sobrescreve banco existente. Para recriá-lo deliberadamente, faça backup e execute:

```powershell
Copy-Item data/sample/sales.db "$HOME\Downloads\sales-backup.db"
python -m bi_multiagent.data.seed --force
```

Para usar um banco separado em benchmark:

```powershell
python -m bi_multiagent.data.seed --output .\data\sample\sales_benchmark_v2.db
$env:BI_MULTIAGENT_DATABASE_PATH = (Resolve-Path .\data\sample\sales_benchmark_v2.db).Path
```

Para voltar ao caminho padrão na sessão atual:

```powershell
Remove-Item Env:BI_MULTIAGENT_DATABASE_PATH -ErrorAction SilentlyContinue
```

## 6. Executar

Agente individual (ReAct):

```powershell
python -m bi_multiagent.agents.cli --thread-id single-demo
```

Supervisor multiagente (requer PostgreSQL configurado e tabelas de checkpoint):

```powershell
python -m bi_multiagent.agents.multiagent_cli --thread-id multiagent-demo
```

Para testar HITL, peça explicitamente para salvar o relatório em arquivo. O CLI deve pausar, solicitar aprovação/rejeição e só então executar ou rejeitar a gravação.

## 7. Recuperação Supabase PGVector

FAISS continua sendo o backend padrão. A recuperação Supabase é ativada somente quando:

```dotenv
KNOWLEDGE_VECTOR_BACKEND=supabase
```

Crie a extensão/tabela/índices no SQL Editor do Supabase usando:

`supabase/migrations/001_business_knowledge_pgvector.sql`

Depois execute:

```powershell
python -X utf8 .\scripts\check_supabase_pgvector.py
python -X utf8 .\scripts\load_supabase_knowledge_base.py
python -X utf8 .\scripts/check_supabase_pgvector.py
```

O carregador indexa os Markdown de `data/knowledge_base/`. Se a tabela já tiver conteúdo, ele interrompe para evitar duplicação. `--replace` solicita explicitamente a substituição; o script cria uma tabela de backup e tenta restaurar os dados anteriores caso a nova inserção falhe.

Valide a busca:

```powershell
python -X utf8 -c "from bi_multiagent.retrieval.supabase_pgvector import search_supabase_knowledge_base; print(search_supabase_knowledge_base('Qual é a fórmula da receita líquida?', k=3))"
```

Para voltar ao FAISS, defina `KNOWLEDGE_VECTOR_BACKEND=faiss`.

## 8. LangGraph Agent Server

O entrypoint `supervisor` está configurado em `langgraph.json` e aponta para `src/bi_multiagent/coordination/deployment.py`.

Para iniciar o servidor local:

```powershell
python -m pip install "langgraph-cli[inmem]"
langgraph dev --no-browser
```

O servidor local normalmente fica em `http://127.0.0.1:2024`. Use o LangGraph Studio para inspecionar o grafo.

A configuração local do Agent Server não é prova de deployment cloud. Para um deployment remoto, revise variáveis/segredos, conectividade com Ollama e PostgreSQL, autenticação e os requisitos da plataforma de destino.

## 9. LangSmith: benchmark e avaliação do Supervisor

Benchmark ReAct vs. Plan-and-Execute:

```powershell
python -X utf8 .\scripts\evaluate_reasoning_architectures.py --dry-run
python -X utf8 .\scripts\evaluate_reasoning_architectures.py
```

A avaliação ponta a ponta do Supervisor usa seis cenários e cinco avaliadores determinísticos:

```powershell
python -X utf8 .\scripts\evaluate_supervisor_langsmith.py --dry-run
python -X utf8 .\scripts\evaluate_supervisor_langsmith.py
```

Casos cobertos: receita líquida total, diferença entre receita bruta/líquida, pergunta combinada de vendas e fórmula, período Selic, regressão de `discount_pct` e exportação com aprovação humana.

Os avaliadores verificam roteamento, política HITL, status da auditoria, consenso e semântica da fração de desconto. Eles medem contratos e controles, não garantem a verdade factual de cada resposta.

Após a execução, para verificar e registrar feedback determinístico em um experimento já existente sem chamar novamente o Ollama:

```powershell
python -X utf8 .\scripts\register_existing_supervisor_feedback_sync.py --project "NOME_EXATO_DO_EXPERIMENTO"
```

O script não usa `trace_id` na criação de feedback, evitando submissão batch em background, e ignora feedbacks que já existem.

Para criar um trace novo do cenário corrigido de `discount_pct`:

```powershell
python -X utf8 .\scripts\trace_discount_pct_incident.py
```

Esse script registra uma execução corrigida. Ele não recria nem substitui um trace histórico de uma falha anterior.

## 10. Testes e qualidade

Com o ambiente virtual ativo:

```powershell
python -m pip check
python -m compileall -q src tests
python -m pytest -q
```

A validação local compartilhada em 16/11/2026, antes desta rodada de revisão, registrou `137 passed`. A suíte precisa ser executada novamente depois de aplicar este patch. O workflow em `.github/workflows/tests.yml` executa instalação, `pip check`, compilação e pytest em pushes e pull requests.

O lint global do legado não está declarado como limpo; execute Ruff nos módulos específicos que estiverem sendo alterados e trate avisos herdados separadamente.

## 11. Limitações conhecidas

1. **Deployment remoto:** o repositório possui configuração e entrypoint do Agent Server, mas isso não comprova um deployment cloud publicado.
2. **Ollama remoto:** `http://localhost:11434` é endereço local; um deployment cloud precisa de endpoint de modelo acessível remotamente e protegido.
3. **Consenso:** o score mede suporte/completude das evidências usando pesos definidos por política; não é consenso semântico entre afirmações nem probabilidade calibrada.
4. **Instruction Fidelity:** as regras são determinísticas e dependem de padrões/termos conhecidos. Casos ambíguos exigem revisão.
5. **A2A:** a integração A2A ainda é separada do fluxo principal do Supervisor e precisa de autenticação, TLS e persistência durável para uso de produção.
6. **Segurança operacional:** o protótipo necessita de controles adicionais de autenticação, autorização, gestão de segredos, monitoramento, limites operacionais e observabilidade antes de uso corporativo.

## 12. Contexto acadêmico

Projeto desenvolvido para a disciplina **Agentes Cognitivos com LangChain [26E4_2]**. Ferramentas de IA foram utilizadas como apoio ao desenvolvimento e à documentação; as decisões, validações e evidências devem ser verificadas pelo autor.
