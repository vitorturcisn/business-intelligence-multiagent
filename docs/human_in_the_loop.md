# Human-in-the-Loop e política de concorrência

## Fluxo de aprovação

O Supervisor identifica pedidos explícitos para salvar ou exportar um
relatório. Após a análise e a síntese, o grafo chama `interrupt()` no nó
`human_approval`.

O payload contém a ação proposta, a solicitação, uma prévia do relatório
e o destino do arquivo.

A decisão humana ? retomada por `Command(resume=...)`.

- Aprovação: o executor grava um arquivo Markdown em
  `outputs/approved_reports/`.
- Rejeição: nenhum arquivo é criado e o resultado registra a rejeição.

O n? interrompido não executa efeitos colaterais antes do `interrupt()`.
O executor valida novamente a aprovação antes de gravar o arquivo.
A escrita ? idempotente para o mesmo `request_id`.

## Persistência

A execução interrompida utiliza o checkpointer PostgreSQL e o mesmo
`thread_id`. Ao reiniciar o CLI, uma interrupção pendente é recuperada
pelo snapshot e pode ser retomada.

## Double-texting

A aplicação utiliza `pg_try_advisory_lock` com uma chave derivada do
`thread_id`. O lock permanece ativo durante o processamento e enquanto
o CLI aguarda a decisão humana.

Uma segunda execução da mesma conversa ? rejeitada com
`DOUBLE_TEXTING_BLOCKED`; ela não ? enfileirada.

## Limite do executor

A ação implementada ? uma gravação local de relatório. Não existe envio
de e-mail, publicação externa ou alteração de sistemas de terceiros.
Essas integrações exigiriam executores próprios, com autorização e
auditoria específicas.
