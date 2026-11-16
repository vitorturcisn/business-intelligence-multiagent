$ErrorActionPreference = 'Stop'

$ProjectRoot = 'C:\Users\vitor\Documents\Projetos\business-intelligence-multiagent'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$CloneRoot = Join-Path $env:TEMP ('business-intelligence-multiagent-clean-' + [guid]::NewGuid().ToString('N'))
$LogPath = Join-Path $HOME 'Downloads\validacao_clone_limpo.txt'

if (-not (Test-Path $ProjectRoot)) { throw "Projeto não encontrado: $ProjectRoot" }
if (-not (Test-Path $Python)) { throw "Python da .venv não encontrado: $Python" }

New-Item -ItemType Directory -Path $CloneRoot -Force | Out-Null
Start-Transcript -Path $LogPath -Force

$OriginalPythonPath = $env:PYTHONPATH
$Succeeded = $false

try {
    Write-Host "========== 1. COPIANDO PROJETO SEM ARTEFATOS LOCAIS ==========" -ForegroundColor Cyan
    Write-Host "Clone temporário: $CloneRoot"

    $ExcludeDirs = @(
        (Join-Path $ProjectRoot '.git'),
        (Join-Path $ProjectRoot '.venv'),
        (Join-Path $ProjectRoot '__pycache__'),
        (Join-Path $ProjectRoot '.pytest_cache'),
        (Join-Path $ProjectRoot '.ruff_cache'),
        (Join-Path $ProjectRoot 'outputs'),
        (Join-Path $ProjectRoot '.secrets'),
        (Join-Path $ProjectRoot '.backups'),
        (Join-Path $ProjectRoot 'data\indexes')
    )

    $RobocopyArgs = @(
        $ProjectRoot, $CloneRoot, '/E', '/R:1', '/W:1', '/NFL', '/NDL', '/NP',
        '/XF', '.env', '*.db', '*.sqlite', '*.sqlite3', '*.pyc', '*.log',
        '/XD'
    ) + $ExcludeDirs

    & robocopy @RobocopyArgs
    $CopyExitCode = $LASTEXITCODE
    if ($CopyExitCode -ge 8) {
        throw "Robocopy falhou com código $CopyExitCode."
    }

    $CloneDb = Join-Path $CloneRoot 'data\sample\sales.db'
    if (Test-Path $CloneDb) {
        throw 'O banco foi copiado indevidamente para o clone de validação.'
    }

    Write-Host '[OK] Clone criado sem .env, .venv, Git, SQLite, índices e outputs.' -ForegroundColor Green

    Write-Host "`n========== 2. TESTANDO CLONE SEM SALES.DB ==========" -ForegroundColor Cyan
    Push-Location $CloneRoot
    try {
        $env:PYTHONPATH = Join-Path $CloneRoot 'src'
        & $Python -m pytest -q
        $PytestExitCode = $LASTEXITCODE
        if ($PytestExitCode -ne 0) {
            throw "Pytest falhou no clone limpo. Exit code: $PytestExitCode"
        }

        Write-Host "`n========== 3. CONFIRMANDO SEED AUTOMÁTICO ==========" -ForegroundColor Cyan
        & $Python -c "from pathlib import Path; from bi_multiagent.settings import DATABASE_PATH; from bi_multiagent.data.seed import has_sales_schema; print('Database:', DATABASE_PATH); print('Exists:', DATABASE_PATH.exists()); print('Valid schema:', has_sales_schema(DATABASE_PATH)); assert DATABASE_PATH.exists() and has_sales_schema(DATABASE_PATH)"
        if ($LASTEXITCODE -ne 0) { throw 'O banco não foi gerado corretamente pelo fixture.' }

        $Succeeded = $true
    }
    finally {
        Pop-Location
        $env:PYTHONPATH = $OriginalPythonPath
    }

    Write-Host "`nVALIDAÇÃO DE CLONE LIMPO: APROVADA." -ForegroundColor Green
}
catch {
    Write-Host "`nFALHA: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "O clone temporário foi preservado para diagnóstico: $CloneRoot" -ForegroundColor Yellow
    throw
}
finally {
    $env:PYTHONPATH = $OriginalPythonPath
    Stop-Transcript | Out-Null

    if ($Succeeded -and (Test-Path $CloneRoot)) {
        Remove-Item -LiteralPath $CloneRoot -Recurse -Force
        Write-Host "Clone temporário removido. Log: $LogPath" -ForegroundColor Green
    }
}
