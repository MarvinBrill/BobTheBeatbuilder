# Prepared reset for the existing local data, October 26, 2026.
# Close Bob before running. Original osu! files and installed runtimes are preserved.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = (Resolve-Path -LiteralPath (Join-Path $repoRoot 'data')).Path
if ((Get-Item -LiteralPath $dataRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'The data directory must not be a link.'
}
$python = Join-Path $dataRoot 'venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'The local Python runtime is missing.' }
if (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in @(8000, 8001) }) {
    throw 'Close the Bob backend on ports 8000/8001 before resetting its data.'
}

$env:PYTHONPATH = Join-Path $repoRoot 'backend'
$env:BOB_DATA = $dataRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = Join-Path $dataRoot 'tmp'
$env:TMP = $env:TEMP

# These five outputs and two uploads were created by the user, not by test runs.
$keepOutputs = @(
    '37375702bc7047dfa4f70c9647be7e3f', 'f490cfcbf7a84ac981630e1db2908cab',
    '045adcb9d8a8452280819d5d59e397eb', 'cde194ddba0b4bffbef9a54f1f065696',
    '6346191d01f8405c8247f67b620c3b2a'
)
$expectedModels = @(
    '04bca6322733445db4cef0c950caac39', '3e332e864d994bd896529e63c8c7bb1c',
    '4d52d35f301f40d9a5e294a2bcf12620', '5681c8fa73764477a2cc07b29801413c',
    '763b185e41664cab91dda21d17dc4be6', '79eef01be98b4616b5092da9ff0462c9',
    '7a6e7b413abb4b328dcd12308b0df21b', '81ba606fe1eb4252ac6cfeeeac3e4a18',
    '85dbe7441a004b429fb437397bb28b24', '875341cb49414055b30a1945b7b29126',
    '97cbf5c19a6c428f82b41612d30a666f', '9ea15b553f254c1d94aa4cc9f638b43c',
    'be270a9192aa495d8e813d7ba1fdc1f1', 'cc3dc95afefc4f17b015ebf7a2bad387',
    'd290e436173f4369b53ac36ad5ac2511'
)
$expectedDatasets = @('12b67baa1a494b02a56cd27e2852c0cf', '5c0e3180f4b841f0b42472ed0942c59f')
$testOutputs = @(
    '0dc2161595f4401cad815e3a9d7b6cf0', '10c8ad49b7484588835bcbb166ecbfe9',
    '1546d9b5f8c24d4c8a2348371daf07e4', '22f80444916c46238c45f79756807d38',
    '30828b02d1024d05abb8cf08f37dfeb1', '498067cfd67748a9a9c92d653038a3c5',
    '7e8bb55d68c545f594abd46772950414', 'a69542ceb80149d5b8c217a56d8624e4',
    'a7519add203a405da50d9631e41a0302', 'b9b8076c617741b59f143e917adf4fc8',
    'f16bf1635ce94e1fa952fdfa4399d5ea'
)

$inspection = @'
import json
from app import store
jobs = store.all_records('job')
if any(job['status'] in ('queued', 'running') for job in jobs):
    raise SystemExit('An active job exists. Finish or cancel it first.')
print(json.dumps({kind: [item['id'] for item in store.all_records(kind)]
                  for kind in ('model', 'dataset', 'output', 'audio')}))
'@ | & $python -
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the data safely.' }
$inventory = $inspection | ConvertFrom-Json

function Assert-KnownIds($actual, $expected, $label) {
    if (@($actual | Where-Object { $_ -notin $expected }).Count) {
        throw "New $label data exists. This prepared reset must be reviewed before deleting it."
    }
}
Assert-KnownIds $inventory.model $expectedModels 'model'
Assert-KnownIds $inventory.dataset $expectedDatasets 'dataset'
Assert-KnownIds $inventory.output ($keepOutputs + $testOutputs) 'output'
Assert-KnownIds $inventory.audio @('9bc00dbb129f4d22b43ea91c7b78fd87', '6b118015ee62470f81d59b871646aed2', 'b0b309ddf3914fb9b2f615465a0366a2') 'audio'

$targets = @()
foreach ($group in @(
    @{ Folder = 'models'; Ids = $expectedModels },
    @{ Folder = 'datasets'; Ids = $expectedDatasets },
    @{ Folder = 'outputs'; Ids = $testOutputs }
)) {
    $folder = Join-Path $dataRoot $group.Folder
    if (!(Test-Path -LiteralPath $folder)) { continue }
    $children = @(Get-ChildItem -LiteralPath $folder -Force)
    $allowed = if ($group.Folder -eq 'outputs') { $keepOutputs + $testOutputs } else { $group.Ids }
    Assert-KnownIds @($children.Name) $allowed $group.Folder
    $targets += $children | Where-Object { $_.Name -in $group.Ids }
}
foreach ($relative in @(
    'tests', 'benchmarks', 'cache/pytest',
    'uploads/b0b309ddf3914fb9b2f615465a0366a2.mp3',
    'probe_phrase.py', 'smoke-directml.py', 'smoke-real.py',
    'verify_cuda_training.py', 'verify_directml_cuda_compat.py',
    'verify_phrase.py', 'verify_review.py', 'studio-preview.png'
)) {
    $candidate = Join-Path $dataRoot $relative
    if (Test-Path -LiteralPath $candidate) { $targets += Get-Item -LiteralPath $candidate }
}
$targets += Get-ChildItem -LiteralPath $dataRoot -Directory | Where-Object { $_.Name -like 'pytest-*' }
$targets += Get-ChildItem -LiteralPath $dataRoot -File | Where-Object { $_.Name -like 'verification-*.json' }
$targets += Get-ChildItem -LiteralPath (Join-Path $dataRoot 'cache') -File | Where-Object { $_.Extension -in @('.npy', '.npz') }
$targets = @($targets | Sort-Object FullName -Unique)

# Validate every resolved target and reject links before any deletion starts.
$totalBytes = 0L
foreach ($target in $targets) {
    $resolved = (Resolve-Path -LiteralPath $target.FullName).Path
    if (!$resolved.StartsWith($dataRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside repository data directory: $resolved"
    }
    $ancestor = $target
    while ($ancestor -and $ancestor.FullName -ne $dataRoot) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Linked path rejected: $($ancestor.FullName)"
        }
        $ancestor = Get-Item -LiteralPath (Split-Path -Parent $ancestor.FullName)
    }
    $children = @(if ($target.PSIsContainer) { Get-ChildItem -LiteralPath $resolved -Recurse -Force })
    if ($children | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) {
        throw "Linked child rejected: $resolved"
    }
    if ($target.PSIsContainer) {
        $totalBytes += ($children | Where-Object { !$_.PSIsContainer } | Measure-Object Length -Sum).Sum
    } else { $totalBytes += $target.Length }
}
Write-Host ("Removing {0} checked targets ({1:N2} GiB)." -f $targets.Count, ($totalBytes / 1GB))
foreach ($target in $targets) { Remove-Item -LiteralPath $target.FullName -Recurse -Force }

# Remove records only after all corresponding files have been removed successfully.
@'
from app import store
keep = {'37375702bc7047dfa4f70c9647be7e3f', 'f490cfcbf7a84ac981630e1db2908cab',
        '045adcb9d8a8452280819d5d59e397eb', 'cde194ddba0b4bffbef9a54f1f065696',
        '6346191d01f8405c8247f67b620c3b2a'}
jobs = store.all_records('job')
outputs = store.all_records('output')
with store.connect() as connection:
    connection.execute("DELETE FROM records WHERE kind IN ('model','dataset')")
    connection.execute("DELETE FROM records WHERE kind='audio' AND id=?", ('b0b309ddf3914fb9b2f615465a0366a2',))
    connection.executemany("DELETE FROM records WHERE kind='output' AND id=?",
                           [(item['id'],) for item in outputs if item['id'] not in keep])
    connection.executemany("DELETE FROM records WHERE kind='job' AND id=?",
                           [(job['id'],) for job in jobs
                            if (job.get('result') or {}).get('output_id') not in keep])
assert store.count('model') == store.count('dataset') == 0
print('Reset complete. Models and datasets are empty. Original maps and user outputs are preserved.')
'@ | & $python -
if ($LASTEXITCODE -ne 0) { throw 'Files were removed, but database cleanup failed. Run this script again before starting Bob.' }
