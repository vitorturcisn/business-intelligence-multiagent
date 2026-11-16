# Etapa 5 - LangGraph Agent Server + Supabase PGVector

## Decisões de implementação

- O grafo exportado para Agent Server é compilado sem `InMemorySaver`; o Agent Server injeta seu próprio checkpointer no ambiente de deployment.
- O CLI multiagente local permanece com `PostgresSaver` e advisory lock.
- FAISS continua sendo o backend padrão (`KNOWLEDGE_VECTOR_BACKEND=faiss`).
- Supabase PGVector é ativado somente com `KNOWLEDGE_VECTOR_BACKEND=supabase`.
- A dimensão de embedding é fixa em 384, correspondente ao modelo `paraphrase-multilingual-MiniLM-L12-v2`.
- O entrypoint Supabase importa `HuggingFaceEmbeddings` diretamente; não carrega o módulo FAISS. `langgraph.json` declara `langchain-postgres`, `langchain-huggingface[full]` e `sentence-transformers` para o runtime do Agent Server.

## 1. Instalar dependências

Na raiz do projeto, com `.venv` ativa:

```powershell
python -m pip install -e ".[dev,retrieval]"
```

Confirme:

```powershell
langgraph --help
python -c "from langchain_postgres import PGEngine, PGVectorStore; print('PGVectorStore OK')"
```

## 2. Configurar Supabase

No Dashboard Supabase, copie a URI PostgreSQL do projeto em **Project Settings > Database > Connection string > URI**. Prefira a conexão Direct ou Session Pooler compatível com conexões persistentes; use SSL.

Adicione ao `.env` local (não ao Git):

```dotenv
SUPABASE_DB_URI=postgresql://USER:PASSWORD@HOST:5432/postgres?sslmode=require
SUPABASE_SSLMODE=require
SUPABASE_VECTOR_SCHEMA=public
SUPABASE_VECTOR_TABLE=business_knowledge_vectors
SUPABASE_VECTOR_SIZE=384
KNOWLEDGE_VECTOR_BACKEND=faiss
```

Não use `SUPABASE_KEY` no lugar de `SUPABASE_DB_URI`: a chave REST não é uma URI PostgreSQL.

## 3. Criar estrutura vetorial

Abra Supabase > SQL Editor e execute `supabase/migrations/001_business_knowledge_pgvector.sql`.

A migration habilita pgvector quando ainda não existe, respeita o schema no qual a extensão já está instalada e cria a tabela em `public`, com índices HNSW/GIN. Se a extensão já existir em `public`, a coluna vetorial e o operador de índice usam esse schema.

## 4. Testar conexão e carregar documentos

```powershell
python -X utf8 .\scripts\check_supabase_pgvector.py
python -X utf8 .\scripts\load_supabase_knowledge_base.py
python -X utf8 .\scripts\check_supabase_pgvector.py
```

O carregador não apaga registros existentes por padrão. Se a tabela já tiver dados e você realmente desejar reconstruí-la, use `--replace` explicitamente. Antes da substituição, o carregador cria uma tabela de backup; se a inserção falhar, ele tenta restaurar os registros anteriores. Se a restauração automática também falhar, o nome da tabela de backup é exibido para recuperação manual.

## 5. Ativar Supabase no Knowledge Agent

Só depois de confirmar a tabela e os documentos:

```powershell
# Edite somente o .env local e altere:
# KNOWLEDGE_VECTOR_BACKEND=supabase

python -X utf8 -m pytest -q tests\unit\test_supabase_pgvector_config.py tests\unit\test_multiagent.py tests\unit\test_instruction_fidelity_audit.py
```

Teste a busca vetorial diretamente e confira se ela retorna conteúdo e `Fonte:`:

```powershell
python -X utf8 -c "from bi_multiagent.retrieval.supabase_pgvector import search_supabase_knowledge_base; print(search_supabase_knowledge_base('Qual é a fórmula da receita líquida?', k=3))"
```

Depois execute uma pergunta conceitual pelo CLI multiagente e confirme que o resultado do Knowledge Analyst inclui as fontes Markdown recuperadas:

```powershell
python -X utf8 -m bi_multiagent.agents.multiagent_cli --thread-id supabase-knowledge-smoke
```

Digite a pergunta conceitual no prompt e encerre com `sair`.

Para voltar ao FAISS, defina `KNOWLEDGE_VECTOR_BACKEND=faiss`.

## 6. Validar LangGraph Agent Server

```powershell
python -m pip install "langgraph-cli[inmem]"
langgraph dev --no-browser
```

O servidor local fica em `http://127.0.0.1:2024`. Use o LangGraph Studio para abrir o grafo `supervisor`, executar uma pergunta e inspecionar o trace.

O grafo `supervisor` está em `langgraph.json`. O módulo `deployment.py` usa `build_multiagent_graph(use_in_memory_saver=False)`: não injete manualmente PostgresSaver neste entrypoint, pois o Agent Server administra a persistência do runtime. O `langgraph.json` declara explicitamente as dependências necessárias ao backend Supabase/embeddings; o CLI multiagente fora do Agent Server continua usando o PostgresSaver do projeto.

Opcional, para stack de validação mais próxima de produção (requer Docker):

```powershell
langgraph up
```

Para deployment cloud, após validar `langgraph dev`, use `langgraph deploy` apenas se a organização/plano tiver Agent Server Deployment habilitado e o endpoint do modelo Ollama for acessível a partir do ambiente remoto. `OLLAMA_BASE_URL=http://localhost:11434` aponta para o próprio host do processo e não será acessível por um deployment cloud remoto sem hospedar/expor um endpoint adequado.

**Segredos:** `langgraph.json` referencia `.env` para facilitar desenvolvimento local. Antes de `langgraph deploy`, revise esse arquivo porque o CLI pode enviar as variáveis dele como secrets do deployment. Não envie credenciais locais de Postgres nem URLs inacessíveis ao ambiente remoto.

## Critérios de aceite

- `langgraph dev --no-browser` sobe sem erro e carrega o grafo `supervisor`.
- Uma pergunta de vendas chama Sales; uma pergunta conceitual chama Knowledge.
- O trace no Studio mostra nodes do Supervisor, especialistas, consenso ponderado e Instruction Fidelity Audit.
- A migration cria tabela e extensão; o loader importa os quatro Markdown; uma consulta retorna fonte e conteúdo.
- Com `KNOWLEDGE_VECTOR_BACKEND=supabase`, o Knowledge Agent consulta PGVector; com `faiss`, mantém o caminho local.
- HITL pausa antes do arquivo e retoma com estado do Agent Server.

A existência dos arquivos de configuração não significa que houve deployment remoto. Só declare deploy concluído depois de `langgraph dev` ou `langgraph up` passar e, se for requisito, depois de existir deployment publicado com URL/ID verificáveis.
