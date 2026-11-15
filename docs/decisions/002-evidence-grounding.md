# ADR 002 - Recuperação explícita de evidências e limites de fontes

## Contexto

A execução com Qwen3 apresentou indicadores e fontes que não existiam
na base documental. O modelo também não identificou corretamente o nome
da tabela, apesar de o schema estar disponível.

## Evidências verificadas

- O banco SQLite possui uma tabela principal chamada `sales`.
- O glossário documenta receita bruta, descontos, receita líquida,
  quantidade vendida, número de pedidos, receita por categoria e
  comparação entre períodos.
- O dicionário define as fórmulas de receita e desconto.
- A fonte externa documentada é a série 432 do BCB para meta Selic.
- CAC, lucro bruto e série 433 não estão documentados como indicadores
  comerciais disponíveis neste projeto.

## Decisão

Adicionar um nó `retrieve_context` antes do nó `agent`.

O nó recupera:
- Schema para perguntas sobre tabelas e dados de vendas.
- Documentos completos relevantes para perguntas de negócio.
- Referências explícitas aos arquivos de origem.

O ciclo ReAct com ToolNode e tools_condition permanece ativo.

## Limites de fontes

- SQLite: valores e métricas comerciais.
- Markdown: definições, fórmulas e regras de negócio.
- BCB SGS 432: meta Selic.
- Outras séries não documentadas não são autorizadas.

## Próximos passos

- Avaliar a qualidade do retrieval com dataset específico.
- Comparar a busca lexical com FAISS.
- Na fase de produção, avaliar Supabase PGVector.
- Incluir avaliação factual e rastreamento no LangSmith.