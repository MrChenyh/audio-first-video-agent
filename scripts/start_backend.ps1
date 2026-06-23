$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = $env:PYTHON

if (-not $Python) {
  $Python = "python"
}

Set-Location $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "backend"
& $Python -m uvicorn app.main:app --reload --app-dir backend --host 127.0.0.1 --port 8000
