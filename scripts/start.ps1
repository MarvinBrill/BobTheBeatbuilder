param([ValidateSet('cuda','directml','cpu')][string]$Device = 'cuda', [string]$OsuPath = 'D:/osu! lazer')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$python = Join-Path $repoRoot "data/venv-$Device/Scripts/python.exe"
$legacyVenv = $false
if (!(Test-Path -LiteralPath $python) -and (Test-Path data/venv/Scripts/python.exe)) {
    $python = Join-Path $repoRoot 'data/venv/Scripts/python.exe'
    $legacyVenv = $true
}

# After a git pull/merge the built frontend and the .NET tools can be older than their
# sources. Rebuild exactly the parts whose sources are newer than the build output.
function Newest-Source([string[]]$Paths, [string[]]$Extensions) {
    $files = foreach ($path in $Paths) {
        if (Test-Path -LiteralPath $path -PathType Leaf) { Get-Item -LiteralPath $path }
        elseif (Test-Path -LiteralPath $path) {
            Get-ChildItem -LiteralPath $path -Recurse -File | Where-Object {
                $_.FullName -notmatch '[\\/](bin|obj|node_modules|dist)[\\/]' -and
                (!$Extensions -or $Extensions -contains $_.Extension.ToLowerInvariant())
            }
        }
    }
    $latest = $files | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if ($latest) { return $latest.LastWriteTimeUtc }
    return $null
}
function Is-Stale([string]$Output, [string[]]$Sources, [string[]]$Extensions = @()) {
    if (!(Test-Path -LiteralPath $Output)) { return $true }
    $newest = Newest-Source $Sources $Extensions
    return [bool]($newest -and $newest -gt (Get-Item -LiteralPath $Output).LastWriteTimeUtc)
}
function Marker-Differs([string]$Marker, [string]$Source) {
    if (!(Test-Path -LiteralPath $Marker)) { return $true }
    $expected = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash
    return (Get-Content -LiteralPath $Marker -Raw).Trim() -ne $expected
}

$steps = @()
if (!(Test-Path -LiteralPath $python) -or
    (!$legacyVenv -and (Marker-Differs "data/venv-$Device/.bob-requirements" 'backend/requirements.txt'))) {
    $steps += 'python'
}
if (Is-Stale 'data/tools/exporter/LazerReader.dll' @('exporter') @('.cs','.csproj','.xml','.xsd')) {
    $steps += 'exporter'
}
if (Is-Stale 'data/tools/difficulty/BobDifficulty.dll' @('difficulty') @('.cs','.csproj')) {
    $steps += 'difficulty'
}
if (!(Test-Path frontend/node_modules) -or
    (Marker-Differs 'data/frontend-runtime/.bob-package-lock' 'frontend/package-lock.json')) {
    $steps += 'frontend-deps'
}
$frontendSources = @(
    'frontend/src', 'frontend/index.html', 'frontend/package.json',
    'frontend/package-lock.json', 'frontend/vite.config.ts', 'frontend/tsconfig.json'
)
if (($steps -contains 'frontend-deps') -or (Is-Stale 'frontend/dist/index.html' $frontendSources)) {
    $steps += 'frontend'
}
if ($steps.Count -gt 0) {
    Write-Host "Updating after source changes: $($steps -join ', ')"
    & "$PSScriptRoot/setup.ps1" -Device $Device -Steps $steps
    if ($steps -contains 'python') {
        $python = Join-Path $repoRoot "data/venv-$Device/Scripts/python.exe"
    }
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
