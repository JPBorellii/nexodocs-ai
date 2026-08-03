Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

Write-Host "1/26 Ruff format check"
& uv run --locked ruff format --check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "2/26 Ruff lint"
& uv run --locked ruff check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "3/26 Pyright"
& uv run --locked pyright
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "4/26 Pytest with coverage"
& uv run --locked pytest
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "5/26 Coverage report"
& uv run --locked coverage report --show-missing --fail-under=80
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "6/26 Knowledge base synchronization check"
& uv run --locked python scripts/generate_knowledge_base.py --check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "7/26 Knowledge base validation"
& uv run --locked python scripts/validate_knowledge_base.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "8/26 Processed knowledge base synchronization check"
& uv run --locked python scripts/process_knowledge_base.py --check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "9/26 Processed knowledge base validation"
& uv run --locked python scripts/validate_processed_knowledge_base.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Write-Host "10/26 Vector index plan synchronization check"
& uv run --locked python scripts/index_knowledge_base.py plan --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "11/26 Offline retrieval evaluation"
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

Write-Host "12/26 RAG contracts validation"
& uv run --locked python scripts/validate_rag_contracts.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "13/26 Offline grounded-answer evaluation"
$previousRagAppEnvironment = [Environment]::GetEnvironmentVariable("APP_ENV", "Process")
$ragEvaluationExitCode = 0
try { [Environment]::SetEnvironmentVariable("APP_ENV", "test", "Process"); & uv run --locked python scripts/evaluate_rag.py; $ragEvaluationExitCode = $LASTEXITCODE }
finally { [Environment]::SetEnvironmentVariable("APP_ENV", $previousRagAppEnvironment, "Process") }
if ($ragEvaluationExitCode -ne 0) { exit $ragEvaluationExitCode }

Write-Host "14/26 Real execution observability contracts"
& uv run --locked python scripts/validate_execution_observability.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "15/26 Frozen retrieval threshold policy"
& uv run --locked python scripts/validate_retrieval_threshold_policy.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "16/26 Versioned retrieval holdout result"
& uv run --locked python scripts/validate_retrieval_holdout_result.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "17/26 Full RAG holdout fixture R01"
& uv run --locked python scripts/validate_full_rag_holdout_fixture.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "18/26 Full RAG holdout R01 technical incident"
& uv run --locked python scripts/validate_full_rag_holdout_r01_technical_incident.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "19/26 OpenAI Structured Outputs schema"
& uv run --locked python scripts/validate_openai_structured_output_schema.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "20/26 Full RAG holdout fixture R02"
& uv run --locked python scripts/validate_full_rag_holdout_fixture.py --attempt r02
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "21/26 Full RAG system freeze R02 at its historical commit"
& uv run --locked python scripts/validate_full_rag_system_freeze.py --attempt r02 --source-ref 5db6e1714bd67bbc05000c0fe14b26f06e312f1c
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "22/26 Full RAG holdout R02 adjudication"
& uv run --locked python scripts/validate_full_rag_holdout_r02_adjudication.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "23/26 Sanitized grounding error codes"
& uv run --locked python scripts/validate_grounding_error_codes.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "24/26 Grounding diagnostic D03 contract"
& uv run --locked python scripts/validate_grounding_diagnostic.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "25/26 Evaluation oracle corrections"
& uv run --locked python scripts/validate_evaluation_oracle_corrections.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "26/26 Consolidated grounding diagnostic D03 result"
& uv run --locked python scripts/validate_grounding_diagnostic_d03_result.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
