# Etapa 6 - LangSmith do Supervisor

## Objetivo

Gerar traces de ponta a ponta do Supervisor com roteamento, subgrafos especialistas, AgentTool, consenso ponderado, Instruction Fidelity Audit e interrupção HITL. O experimento usa seis casos e cinco avaliadores determinísticos.

## Configuração local

No `.env` local (não versionar a chave):

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=business-intelligence-supervisor
# Opcional, quando sua organização usa API keys scoped a workspace:
LANGSMITH_WORKSPACE_ID=
```

Use a chave LangSmith correta e não compartilhe o valor no terminal ou no chat. Para chaves org-scoped, informe `LANGSMITH_WORKSPACE_ID`; o script passa esse ID explicitamente ao cliente LangSmith.

## 1. Dry-run

```powershell
python -X utf8 .\scripts\evaluate_supervisor_langsmith.py --dry-run
```

Não cria dataset remoto nem chama o modelo.

## 2. Avaliação do Supervisor

Confirme que Ollama e `qwen3:1.7b` estão ativos:

```powershell
ollama list
```

Depois rode:

```powershell
python -X utf8 .\scripts\evaluate_supervisor_langsmith.py
```

O script cria/usa `bi-multiagent-supervisor-evaluation-v1`, executa seis cenários e grava `outputs/evaluation/supervisor_langsmith_manifest.json`. O dataset não é sobrescrito quando já existe: se os exemplos divergirem, crie uma nova versão do nome do dataset.

## 3. Evidência regressiva do incidente discount_pct

O dicionário de dados declara que `discount_pct` é fração decimal (0.10 = 10%) e não deve ser dividido por 100. A fórmula correta da receita líquida é `quantity * unit_price * (1 - discount_pct)`.

Execute:

```powershell
python -X utf8 .\scripts\trace_discount_pct_incident.py
```

Isso cria um trace novo no projeto `business-intelligence-supervisor`, tagueado como `incident`, `regression` e `discount_pct`, e salva `outputs/evaluation/discount_pct_incident_trace_evidence.json`.

Importante: o script registra uma execução regressiva corrigida. Ele não recria o trace histórico original. Se houver um run ID original de antes da correção, anote-o no relatório final como evidência histórica adicional. Se não existir, declare explicitamente essa limitação.

## 4. O que conferir na UI

No projeto `business-intelligence-supervisor`:

- abrir o experimento `supervisor-e2e-*`;
- verificar que os seis casos foram executados;
- abrir um caso e confirmar o root trace e seus child runs;
- inspecionar os nós do Supervisor, Sales/Knowledge/Economic, auditoria, consenso e HITL;
- confirmar metadados e tags `supervisor`, `multi-agent`, `phase6`;
- abrir `supervisor-discount-pct-regression` e validar a evidência do SQL correto.

## Avaliadores

- `supervisor_routing_fidelity`: compara agentes esperados e selecionados.
- `supervisor_hitl_policy_fidelity`: verifica pausa para exportação explícita.
- `instruction_fidelity_audit_passed`: exige status `passed` no auditor.
- `weighted_evidence_consensus_accepted`: exige consenso `accepted`.
- `discount_pct_fraction_semantics`: verifica a fórmula com fração e ausência de divisão por 100 no SQL do agente de vendas.

Esses avaliadores são checagens determinísticas do contrato do sistema; não são garantia de verdade factual.
