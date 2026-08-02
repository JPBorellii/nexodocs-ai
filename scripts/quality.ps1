Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

Write-Host "1/14 Ruff format check"
& uv run --locked ruff format --check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "2/14 Ruff lint"
& uv run --locked ruff check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "3/14 Pyright"
& uv run --locked pyright
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "4/14 Pytest with coverage"
& uv run --locked pytest
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "5/14 Coverage report"
& uv run --locked coverage report --show-missing --fail-under=80
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "6/14 Knowledge base synchronization check"
& uv run --locked python scripts/generate_knowledge_base.py --check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "7/14 Knowledge base validation"
& uv run --locked python scripts/validate_knowledge_base.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "8/14 Processed knowledge base synchronization check"
& uv run --locked python scripts/process_knowledge_base.py --check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "9/14 Processed knowledge base validation"
& uv run --locked python scripts/validate_processed_knowledge_base.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Write-Host "10/14 Vector index plan synchronization check"
& uv run --locked python scripts/index_knowledge_base.py plan --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "11/14 Offline retrieval evaluation"
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

Write-Host "12/14 RAG contracts validation"
& uv run --locked python scripts/validate_rag_contracts.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "13/14 Offline grounded-answer evaluation"
$previousRagAppEnvironment = [Environment]::GetEnvironmentVariable("APP_ENV", "Process")
$ragEvaluationExitCode = 0
try { [Environment]::SetEnvironmentVariable("APP_ENV", "test", "Process"); & uv run --locked python scripts/evaluate_rag.py; $ragEvaluationExitCode = $LASTEXITCODE }
finally { [Environment]::SetEnvironmentVariable("APP_ENV", $previousRagAppEnvironment, "Process") }
if ($ragEvaluationExitCode -ne 0) { exit $ragEvaluationExitCode }

Write-Host "14/14 Real execution observability contracts"
& uv run --locked python scripts/validate_execution_observability.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
