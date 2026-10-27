param([ValidateSet('cuda','directml','cpu')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

foreach ($folder in @('tmp','cache/pip','cache/npm','cache/nuget-http','nuget','dotnet-home','frontend-runtime')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "data/$folder") | Out-Null
}
$env:TEMP = Join-Path $repoRoot 'data/tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $repoRoot 'data/cache/pip'
$env:npm_config_cache = Join-Path $repoRoot 'data/cache/npm'
$env:DOTNET_CLI_HOME = Join-Path $repoRoot 'data/dotnet-home'
$env:NUGET_PACKAGES = Join-Path $repoRoot 'data/nuget'
$env:NUGET_HTTP_CACHE_PATH = Join-Path $repoRoot 'data/cache/nuget-http'
$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'
$env:DOTNET_GENERATE_ASPNET_CERTIFICATE = 'false'
$env:PYTHONDONTWRITEBYTECODE = '1'

function Run-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

foreach ($tool in @('python','node','npm.cmd','dotnet','ffmpeg')) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) { throw "Missing prerequisite: $tool. See README.md. No automatic system-wide installs are performed." }
}
Run-Checked 'python' @('-c','import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Python 3.10-3.12 required"')
$venv = Join-Path $repoRoot "data/venv-$Device"
if (!(Test-Path -LiteralPath "$venv/Scripts/python.exe")) { Run-Checked 'python' @('-m','venv',$venv) }
$python = Join-Path $venv 'Scripts/python.exe'
if ($Device -eq 'cuda') {
    Run-Checked $python @('-m','pip','install','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cu128')
} elseif ($Device -eq 'directml') {
    Run-Checked $python @('-m','pip','install','torch==2.4.1','torchvision==0.19.1','--index-url','https://download.pytorch.org/whl/cpu')
    Run-Checked $python @('-m','pip','install','torch-directml==0.2.5.dev240914')
} else {
    Run-Checked $python @('-m','pip','install','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cpu')
}
Run-Checked $python @('-m','pip','install','-r','backend/requirements.txt')
Run-Checked 'dotnet' @('publish','exporter/LazerReader.csproj','-c','Release','-o','data/tools/exporter',"-p:BaseIntermediateOutputPath=$repoRoot/data/obj/exporter/","-p:BaseOutputPath=$repoRoot/data/bin/exporter/")

Copy-Item frontend/package.json,frontend/package-lock.json data/frontend-runtime/
Push-Location data/frontend-runtime
try { Run-Checked 'npm.cmd' @('ci','--no-audit','--no-fund') } finally { Pop-Location }
if (!(Test-Path frontend/node_modules)) {
    New-Item -ItemType Junction -Path frontend/node_modules -Target "$repoRoot/data/frontend-runtime/node_modules" | Out-Null
}
Push-Location frontend
try { Run-Checked 'npm.cmd' @('run','build') } finally { Pop-Location }
Write-Host "Setup finished. Run ./scripts/start.ps1 -Device $Device"
