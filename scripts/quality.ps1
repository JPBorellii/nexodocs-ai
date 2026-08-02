Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

Write-Host "1/15 Ruff format check"
& uv run --locked ruff format --check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "2/15 Ruff lint"
& uv run --locked ruff check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "3/15 Pyright"
& uv run --locked pyright
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "4/15 Pytest with coverage"
& uv run --locked pytest
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "5/15 Coverage report"
& uv run --locked coverage report --show-missing --fail-under=80
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "6/15 Knowledge base synchronization check"
& uv run --locked python scripts/generate_knowledge_base.py --check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "7/15 Knowledge base validation"
& uv run --locked python scripts/validate_knowledge_base.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "8/15 Processed knowledge base synchronization check"
& uv run --locked python scripts/process_knowledge_base.py --check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "9/15 Processed knowledge base validation"
& uv run --locked python scripts/validate_processed_knowledge_base.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Write-Host "10/15 Vector index plan synchronization check"
& uv run --locked python scripts/index_knowledge_base.py plan --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "11/15 Offline retrieval evaluation"
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

Write-Host "12/15 RAG contracts validation"
& uv run --locked python scripts/validate_rag_contracts.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "13/15 Offline grounded-answer evaluation"
$previousRagAppEnvironment = [Environment]::GetEnvironmentVariable("APP_ENV", "Process")
$ragEvaluationExitCode = 0
try { [Environment]::SetEnvironmentVariable("APP_ENV", "test", "Process"); & uv run --locked python scripts/evaluate_rag.py; $ragEvaluationExitCode = $LASTEXITCODE }
finally { [Environment]::SetEnvironmentVariable("APP_ENV", $previousRagAppEnvironment, "Process") }
if ($ragEvaluationExitCode -ne 0) { exit $ragEvaluationExitCode }

Write-Host "14/15 Real execution observability contracts"
& uv run --locked python scripts/validate_execution_observability.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "15/15 Frozen retrieval threshold policy"
& uv run --locked python scripts/validate_retrieval_threshold_policy.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
