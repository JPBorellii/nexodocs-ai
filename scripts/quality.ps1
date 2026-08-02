Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

Write-Host "1/11 Ruff format check"
& uv run --locked ruff format --check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "2/11 Ruff lint"
& uv run --locked ruff check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "3/11 Pyright"
& uv run --locked pyright
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "4/11 Pytest with coverage"
& uv run --locked pytest
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "5/11 Coverage report"
& uv run --locked coverage report --show-missing --fail-under=80
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "6/11 Knowledge base synchronization check"
& uv run --locked python scripts/generate_knowledge_base.py --check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "7/11 Knowledge base validation"
& uv run --locked python scripts/validate_knowledge_base.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "8/11 Processed knowledge base synchronization check"
& uv run --locked python scripts/process_knowledge_base.py --check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "9/11 Processed knowledge base validation"
& uv run --locked python scripts/validate_processed_knowledge_base.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Write-Host "10/11 Vector index plan synchronization check"
& uv run --locked python scripts/index_knowledge_base.py plan --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "11/11 Offline retrieval evaluation"
$previousAppEnvironment = [Environment]::GetEnvironmentVariable("APP_ENV", "Process")
$evaluationExitCode = 0
try {
    [Environment]::SetEnvironmentVariable("APP_ENV", "test", "Process")
    & uv run --locked python scripts/evaluate_retrieval.py
    $evaluationExitCode = $LASTEXITCODE
}
finally {
    [Environment]::SetEnvironmentVariable("APP_ENV", $previousAppEnvironment, "Process")
}
if ($evaluationExitCode -ne 0) { exit $evaluationExitCode }
