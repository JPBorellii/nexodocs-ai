Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

Write-Host "Checking uv availability..."
if ($null -eq (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv 0.11.32 is required but was not found on PATH."
}

$expectedUvVersion = "0.11.32"
$uvVersion = & uv --version
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $uvVersion.StartsWith("uv $expectedUvVersion ")) {
    throw "Expected uv $expectedUvVersion, found: $uvVersion"
}

Write-Host "Ensuring Python 3.14 is available..."
& uv python install 3.14
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Synchronizing the locked development environment..."
& uv sync --locked --dev
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$environmentPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $environmentPython)) {
    throw "The synchronized virtual environment does not contain Python."
}

$pythonVersion = & $environmentPython --version
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $pythonVersion.StartsWith("Python 3.14.")) {
    throw "Expected Python 3.14 in .venv, found: $pythonVersion"
}

Write-Host "Bootstrap completed with $pythonVersion."
