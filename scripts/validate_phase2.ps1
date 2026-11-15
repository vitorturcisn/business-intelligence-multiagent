$ErrorActionPreference = "Stop"

Write-Host "`n========== 1. IMPORTS ==========" -ForegroundColor Cyan

python -c "from bi_multiagent.coordination.supervisor_graph import build_multiagent_graph; print('Supervisor OK')"
if ($LASTEXITCODE -ne 0) { throw "Falha nos imports." }

Write-Host "`n========== 2. TESTES ==========" -ForegroundColor Cyan

python -m pytest -v
if ($LASTEXITCODE -ne 0) { throw "Testes falharam." }

Write-Host "`n========== 3. RUFF ==========" -ForegroundColor Cyan

ruff check src tests
if ($LASTEXITCODE -ne 0) { throw "Ruff falhou." }

Write-Host "`n========== 4. COMPILAÇÃO ==========" -ForegroundColor Cyan

python -m compileall -q src
if ($LASTEXITCODE -ne 0) { throw "Compilação falhou." }

Write-Host "`n========== 5. DEPENDÊNCIAS ==========" -ForegroundColor Cyan

python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Pip check falhou." }

Write-Host "`n========== 6. DIAGRAMA DO GRAFO ==========" -ForegroundColor Cyan

python -c "from bi_multiagent.coordination.supervisor_graph import build_multiagent_graph; print(build_multiagent_graph().get_graph().draw_mermaid())"
if ($LASTEXITCODE -ne 0) { throw "Falha ao gerar diagrama." }

Write-Host "`nVALIDAÇÃO COMPLETA: TODAS AS ETAPAS APROVADAS." -ForegroundColor Green
