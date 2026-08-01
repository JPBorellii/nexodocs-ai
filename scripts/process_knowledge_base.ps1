Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

$uvVersion = & uv --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not $uvVersion.StartsWith("uv 0.11.32 ")) { throw "Expected uv 0.11.32, found: $uvVersion" }

& uv run --locked python scripts/process_knowledge_base.py --write
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& uv run --locked python scripts/validate_processed_knowledge_base.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
