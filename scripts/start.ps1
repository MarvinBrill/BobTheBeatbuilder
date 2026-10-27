param([ValidateSet('cuda','directml','cpu')][string]$Device = 'cuda', [string]$OsuPath = 'D:/osu! lazer')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$python = Join-Path $repoRoot "data/venv-$Device/Scripts/python.exe"
if (!(Test-Path -LiteralPath $python) -and (Test-Path data/venv/Scripts/python.exe)) {
    $python = Join-Path $repoRoot 'data/venv/Scripts/python.exe'
}
if (!(Test-Path -LiteralPath $python) -or !(Test-Path frontend/dist/index.html) -or !(Test-Path data/tools/exporter/LazerReader.dll)) {
    & "$PSScriptRoot/setup.ps1" -Device $Device
}
$env:BOB_DATA = Join-Path $repoRoot 'data'
$env:BOB_DEVICE = $Device
$env:BOB_OSU_PATH = $OsuPath
$env:PYTHONPATH = Join-Path $repoRoot 'backend'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'
$env:DOTNET_GENERATE_ASPNET_CERTIFICATE = 'false'
$env:DOTNET_CLI_HOME = Join-Path $repoRoot 'data/dotnet-home'
$env:TEMP = Join-Path $repoRoot 'data/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
& $python -c 'from app.launch import check_device; check_device()'
if ($LASTEXITCODE -ne 0) { throw 'Compute backend check failed. See the error above.' }
Write-Host 'Bob is available at http://127.0.0.1:8000. Stop with Ctrl+C.'
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
