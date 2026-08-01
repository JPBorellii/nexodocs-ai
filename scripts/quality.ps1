Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

Write-Host "1/5 Ruff format check"
& uv run --locked ruff format --check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "2/5 Ruff lint"
& uv run --locked ruff check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "3/5 Pyright"
& uv run --locked pyright
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "4/5 Pytest with coverage"
& uv run --locked pytest
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "5/5 Coverage report"
& uv run --locked coverage report --show-missing --fail-under=80
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
