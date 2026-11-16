# Etapa 6 - LangSmith do Supervisor

## Objetivo

Gerar traces de ponta a ponta do Supervisor com roteamento, subgrafos especialistas, AgentTool, consenso ponderado, Instruction Fidelity Audit e interrupção HITL. O experimento usa seis casos e cinco avaliadores determinísticos.

## Configuração local

No `.env` local (não versionar a chave):

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=business-intelligence-supervisor
LANGSMITH_WORKSPACE_ID=
```

Use a chave LangSmith correta. Para chaves scoped a workspace, informe `LANGSMITH_WORKSPACE_ID`.

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

Execute:

```powershell
python -X utf8 .\scripts\evaluate_supervisor_langsmith.py
```

O script cria/usa `bi-multiagent-supervisor-evaluation-v1`, executa seis cenários e grava `outputs/evaluation/supervisor_langsmith_manifest.json`. O dataset existente não é sobrescrito quando os exemplos divergem.

## 3. Verificar e registrar feedbacks de um experimento existente

Depois da execução, use o nome exato do experimento exibido no LangSmith:

```powershell
python -X utf8 .\scripts\register_existing_supervisor_feedback_sync.py --project "supervisor-e2e-AAAA-MM-DD-HHMMSS-XXXXXXXX"
```

O script recupera as seis execuções existentes, compara os outputs com os exemplos do dataset e registra os feedbacks determinísticos que estiverem faltando. Não chama Ollama nem refaz inferências. A criação não passa `trace_id`, evitando a submissão batch em background. Execuções que já têm feedback são ignoradas.

A expectativa é de quatro avaliadores aplicados aos seis cenários e um avaliador específico aplicado somente ao caso de regressão de desconto: 25 feedbacks no total.

## 4. Trace novo para a regressão `discount_pct`

```powershell
python -X utf8 .\scripts\trace_discount_pct_incident.py
```

O script executa um novo cenário corrigido, cria trace no projeto LangSmith e salva `outputs/evaluation/discount_pct_incident_trace_evidence.json`.

**Importante:** esse script registra uma execução corrigida. Ele não recria o trace histórico original que apresentou uma falha, nem substitui evidência histórica.

## Avaliadores

- `supervisor_routing_fidelity`: compara agentes esperados e selecionados.
- `supervisor_hitl_policy_fidelity`: verifica pausa para exportação explícita.
- `instruction_fidelity_audit_passed`: exige status `passed` no auditor.
- `weighted_evidence_consensus_accepted`: exige consenso `accepted`.
- `discount_pct_fraction_semantics`: verifica a fórmula com fração e ausência de divisão por 100 no SQL do agente de vendas.

Esses avaliadores são checagens determinísticas do contrato do sistema; não são garantia de verdade factual.

## O que inspecionar na UI

No projeto `business-intelligence-supervisor`, confira:

- o experimento `supervisor-e2e-*`;
- os seis casos e seus root/child runs;
- os nós do Supervisor, especialistas, auditoria, consenso e HITL;
- tags e metadados `supervisor`, `multi-agent`, `phase6`;
- o trace corrigido de `discount_pct`.
