$ErrorActionPreference = "Continue"
$ProjectRoot = "C:\Users\vitor\Documents\Projetos\business-intelligence-multiagent"
$Report = Join-Path $HOME "Downloads\auditoria_etapa1_detalhada.txt"

if (-not (Test-Path $ProjectRoot)) {
    throw "Pasta do projeto não encontrada: $ProjectRoot"
}
Set-Location $ProjectRoot

Start-Transcript -Path $Report -Force

Write-Host "========== 1. GIT STATUS ==========" -ForegroundColor Cyan
git status --short --untracked-files=all

Write-Host "`n========== 2. DIFF DA CONFIGURACAO E README ==========" -ForegroundColor Cyan
git diff -- .env.example README.md

Write-Host "`n========== 3. ENV EXAMPLE ==========" -ForegroundColor Cyan
if (Test-Path ".env.example") {
    Get-Content ".env.example"
    $envExample = Get-Content ".env.example" -Raw
    if ($envExample -match "(?m)^POSTGRES_CHECKPOINTER_URI=") {
        Write-Host "[OK] POSTGRES_CHECKPOINTER_URI documentada" -ForegroundColor Green
    } else {
        Write-Host "[AUSENTE] POSTGRES_CHECKPOINTER_URI" -ForegroundColor Red
    }
} else {
    Write-Host "[AUSENTE] .env.example" -ForegroundColor Red
}

Write-Host "`n========== 4. ARQUIVOS NOVOS ==========" -ForegroundColor Cyan
$files = @(
    "src\bi_multiagent\data\__init__.py",
    "src\bi_multiagent\data\seed.py",
    "tests\conftest.py",
    "tests\unit\test_seed_sales_db.py"
)
foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "[OK] $file" -ForegroundColor Green
        if ($file -ne "src\bi_multiagent\data\__init__.py") {
            Write-Host "--- Conteúdo: $file ---" -ForegroundColor Yellow
            Get-Content $file
        }
    } else {
        Write-Host "[AUSENTE] $file" -ForegroundColor Red
    }
}

Write-Host "`n========== 5. BANCO E SCHEMA ==========" -ForegroundColor Cyan
python -c "from bi_multiagent.settings import DATABASE_PATH; from bi_multiagent.data.seed import has_sales_schema; print('Database:', DATABASE_PATH); print('Exists:', DATABASE_PATH.exists()); print('Valid schema:', has_sales_schema(DATABASE_PATH))"
$schemaExit = $LASTEXITCODE
Write-Host "Exit code schema: $schemaExit"

Write-Host "`n========== 6. TESTE DO SEED ==========" -ForegroundColor Cyan
python -m pytest tests/unit/test_seed_sales_db.py -v
$seedTestExit = $LASTEXITCODE
Write-Host "Exit code teste seed: $seedTestExit"

Write-Host "`n========== 7. SUITE COMPLETA ==========" -ForegroundColor Cyan
python -m pytest -q
$pytestExit = $LASTEXITCODE
Write-Host "Exit code pytest completo: $pytestExit"

Write-Host "`n========== 8. RUFF SOMENTE NOS ARQUIVOS NOVOS ==========" -ForegroundColor Cyan
python -m ruff check src/bi_multiagent/data/seed.py tests/conftest.py tests/unit/test_seed_sales_db.py
$ruffNewExit = $LASTEXITCODE
Write-Host "Exit code Ruff arquivos novos: $ruffNewExit"

Write-Host "`n========== 9. COMPILACAO DOS ARQUIVOS NOVOS ==========" -ForegroundColor Cyan
python -m py_compile src/bi_multiagent/data/seed.py tests/conftest.py tests/unit/test_seed_sales_db.py
$compileExit = $LASTEXITCODE
Write-Host "Exit code compilacao: $compileExit"

Write-Host "`n========== 10. DEPENDENCIAS ==========" -ForegroundColor Cyan
python -m pip check
$pipExit = $LASTEXITCODE
Write-Host "Exit code pip check: $pipExit"

Write-Host "`n========== 11. DIFF CHECK ==========" -ForegroundColor Cyan
git diff --check
$diffExit = $LASTEXITCODE
Write-Host "Exit code diff check: $diffExit"

Write-Host "`n========== RESUMO DOS CODIGOS DE SAIDA ==========" -ForegroundColor Cyan
Write-Host "Schema: $schemaExit"
Write-Host "Teste seed: $seedTestExit"
Write-Host "Pytest completo: $pytestExit"
Write-Host "Ruff arquivos novos: $ruffNewExit"
Write-Host "Compilacao: $compileExit"
Write-Host "Pip check: $pipExit"
Write-Host "Diff check: $diffExit"

Stop-Transcript
Write-Host "`nRelatorio salvo em: $Report" -ForegroundColor Green
