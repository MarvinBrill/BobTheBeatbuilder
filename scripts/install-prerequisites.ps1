# Installs the system prerequisites for BobTheBeatbuilder via winget:
#   Python 3.12, Node.js 22 (incl. npm), .NET SDK 10, FFmpeg
# Already installed and compatible tools are skipped.
#
# Usage (from the repository root, no execution-policy change needed):
#   powershell -ExecutionPolicy Bypass -File .\scripts\install-prerequisites.ps1
param([ValidateSet('cuda','directml','cpu')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'

function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Test-Tool([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-PythonVersion([string]$Exe = 'python') {
    try {
        $out = & $Exe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out -match '^\d+\.\d+$') { return [version]$out }
    } catch {}
    return $null
}

function Test-PythonOk {
    $v = Get-PythonVersion
    return ($v -and $v -ge [version]'3.10' -and $v -le [version]'3.12')
}

function Install-WingetPackage([string]$Id, [string]$Label) {
    Write-Host "`n>> Installiere $Label ($Id) ..." -ForegroundColor Cyan
    & winget install --id $Id --exact --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   winget meldete Exit-Code $LASTEXITCODE fuer $Label - pruefe Ergebnis ..." -ForegroundColor Yellow
    }
    Update-SessionPath
}

if (!(Test-Tool 'winget')) {
    throw "winget wurde nicht gefunden. Installiere/aktualisiere den 'App Installer' aus dem Microsoft Store und starte das Skript erneut."
}

Update-SessionPath
$problems = @()

# --- Python 3.10-3.12 ---------------------------------------------------------
if (Test-PythonOk) {
    Write-Host "OK  Python $(Get-PythonVersion)" -ForegroundColor Green
} else {
    $found = Get-PythonVersion
    if ($found) { Write-Host "Python $found gefunden, benoetigt wird 3.10-3.12." -ForegroundColor Yellow }
    Install-WingetPackage 'Python.Python.3.12' 'Python 3.12'
    if (!(Test-PythonOk)) {
        # Make Python 3.12 the 'python' on PATH (ahead of the Microsoft Store alias or another version).
        $candidates = @(
            (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312'),
            (Join-Path $env:ProgramFiles 'Python312')
        )
        $dir = $candidates | Where-Object { Test-Path (Join-Path $_ 'python.exe') } | Select-Object -First 1
        if ($dir) {
            $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
            $parts = @($dir, (Join-Path $dir 'Scripts')) + (($userPath -split ';') | Where-Object { $_ -and $_ -ne $dir -and $_ -ne (Join-Path $dir 'Scripts') })
            [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User')
            Update-SessionPath
            $env:Path = "$dir;$(Join-Path $dir 'Scripts');$env:Path"
        }
    }
    if (Test-PythonOk) { Write-Host "OK  Python $(Get-PythonVersion)" -ForegroundColor Green }
    else { $problems += 'Python 3.10-3.12 ist nach der Installation nicht als "python" erreichbar.' }
}

# --- Node.js 22 + npm ---------------------------------------------------------
if ((Test-Tool 'node') -and (Test-Tool 'npm.cmd')) {
    Write-Host "OK  Node.js $(& node --version)" -ForegroundColor Green
} else {
    Install-WingetPackage 'OpenJS.NodeJS.22' 'Node.js 22'
    if ((Test-Tool 'node') -and (Test-Tool 'npm.cmd')) { Write-Host "OK  Node.js $(& node --version)" -ForegroundColor Green }
    else { $problems += 'Node.js/npm nicht gefunden.' }
}

# --- .NET SDK 10 --------------------------------------------------------------
function Test-DotnetSdk10 {
    if (!(Test-Tool 'dotnet')) { return $false }
    $sdks = & dotnet --list-sdks 2>$null
    return [bool]($sdks | Where-Object { $_ -match '^10\.' })
}
if (Test-DotnetSdk10) {
    Write-Host "OK  .NET SDK 10" -ForegroundColor Green
} else {
    Install-WingetPackage 'Microsoft.DotNet.SDK.10' '.NET SDK 10'
    if (Test-DotnetSdk10) { Write-Host "OK  .NET SDK 10" -ForegroundColor Green }
    else { $problems += '.NET SDK 10 nicht gefunden.' }
}

# --- FFmpeg -------------------------------------------------------------------
if (Test-Tool 'ffmpeg') {
    Write-Host "OK  FFmpeg" -ForegroundColor Green
} else {
    Install-WingetPackage 'Gyan.FFmpeg' 'FFmpeg'
    if (Test-Tool 'ffmpeg') { Write-Host "OK  FFmpeg" -ForegroundColor Green }
    else { $problems += 'FFmpeg nicht gefunden.' }
}

# --- GPU driver (check only) --------------------------------------------------
if ($Device -eq 'cuda') {
    if (Test-Tool 'nvidia-smi') {
        Write-Host "OK  NVIDIA-Treiber gefunden" -ForegroundColor Green
    } else {
        Write-Host "HINWEIS: nvidia-smi nicht gefunden. Fuer -Device cuda brauchst du einen aktuellen NVIDIA-Treiber (nvidia.com/drivers)." -ForegroundColor Yellow
    }
}

Write-Host ''
if ($problems.Count -gt 0) {
    Write-Host 'Nicht alles konnte eingerichtet werden:' -ForegroundColor Red
    $problems | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host 'Oeffne ein NEUES PowerShell-Fenster und starte dieses Skript noch einmal.' -ForegroundColor Yellow
    exit 1
}

Write-Host 'Alle Voraussetzungen sind installiert.' -ForegroundColor Green
Write-Host 'Oeffne jetzt ein NEUES PowerShell-Fenster (damit PATH aktuell ist) und starte Bob mit:' -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 -Device $Device -OsuPath `"D:/osu! lazer`"" -ForegroundColor White
