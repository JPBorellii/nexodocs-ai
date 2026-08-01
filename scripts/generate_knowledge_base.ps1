Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

$expectedUvVersion = "0.11.32"
$uvVersion = & uv --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not $uvVersion.StartsWith("uv $expectedUvVersion ")) {
    throw "Expected uv $expectedUvVersion, found: $uvVersion"
}

& uv run --locked python scripts/generate_knowledge_base.py --write
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& uv run --locked python scripts/validate_knowledge_base.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
