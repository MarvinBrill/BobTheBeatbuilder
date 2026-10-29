param(
    [ValidateSet('cuda','directml','cpu')][string]$Device = 'cuda',
    # Parts to (re)build. start.ps1 passes only the parts whose sources changed.
    [ValidateSet('python','exporter','difficulty','frontend-deps','frontend')]
    [string[]]$Steps = @('python','exporter','difficulty','frontend-deps','frontend')
)
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

function Source-Hash([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

foreach ($tool in @('python','node','npm.cmd','dotnet','ffmpeg')) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) { throw "Missing prerequisite: $tool. See README.md. No automatic system-wide installs are performed." }
}

$venv = Join-Path $repoRoot "data/venv-$Device"
if ($Steps -contains 'python') {
    Run-Checked 'python' @('-c','import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Python 3.10-3.12 required"')
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
    # start.ps1 compares this to notice changed Python requirements after a git pull.
    Set-Content -LiteralPath "$venv/.bob-requirements" -Value (Source-Hash 'backend/requirements.txt')
}
if ($Steps -contains 'exporter') {
    Run-Checked 'dotnet' @('publish','exporter/LazerReader.csproj','-c','Release','-o','data/tools/exporter',"-p:BaseIntermediateOutputPath=$repoRoot/data/obj/exporter/","-p:BaseOutputPath=$repoRoot/data/bin/exporter/")
}
if ($Steps -contains 'difficulty') {
    # Official osu! star rating calculator (ppy.osu.Game.Rulesets.Osu) for training labels and generated maps.
    Run-Checked 'dotnet' @('publish','difficulty/BobDifficulty.csproj','-c','Release','-o','data/tools/difficulty',"-p:BaseIntermediateOutputPath=$repoRoot/data/obj/difficulty/","-p:BaseOutputPath=$repoRoot/data/bin/difficulty/")
}
if ($Steps -contains 'frontend-deps') {
    Copy-Item frontend/package.json,frontend/package-lock.json data/frontend-runtime/
    Push-Location data/frontend-runtime
    try { Run-Checked 'npm.cmd' @('ci','--no-audit','--no-fund') } finally { Pop-Location }
    if (!(Test-Path frontend/node_modules)) {
        New-Item -ItemType Junction -Path frontend/node_modules -Target "$repoRoot/data/frontend-runtime/node_modules" | Out-Null
    }
    Set-Content -LiteralPath 'data/frontend-runtime/.bob-package-lock' -Value (Source-Hash 'frontend/package-lock.json')
}
if ($Steps -contains 'frontend') {
    Push-Location frontend
    try { Run-Checked 'npm.cmd' @('run','build') } finally { Pop-Location }
}
Write-Host "Setup finished ($($Steps -join ', ')). Run ./scripts/start.ps1 -Device $Device"
