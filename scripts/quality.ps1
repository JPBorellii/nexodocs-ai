Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

Write-Host "1/9 Ruff format check"
& uv run --locked ruff format --check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "2/9 Ruff lint"
& uv run --locked ruff check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "3/9 Pyright"
& uv run --locked pyright
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "4/9 Pytest with coverage"
& uv run --locked pytest
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "5/9 Coverage report"
& uv run --locked coverage report --show-missing --fail-under=80
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "6/9 Knowledge base synchronization check"
& uv run --locked python scripts/generate_knowledge_base.py --check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "7/9 Knowledge base validation"
& uv run --locked python scripts/validate_knowledge_base.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "8/9 Processed knowledge base synchronization check"
& uv run --locked python scripts/process_knowledge_base.py --check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "9/9 Processed knowledge base validation"
& uv run --locked python scripts/validate_processed_knowledge_base.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
