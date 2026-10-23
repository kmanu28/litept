$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-litept\python.exe"
$rawRoot = Join-Path $projectRoot "data\scannet_raw"
$processedRoot = Join-Path $projectRoot "data\scannet_full"
$logPrefix = "[LitePT continuation]"

function Get-CompleteRawSceneCount([string]$Split, [bool]$AnnotationsRequired) {
    $splitRoot = Join-Path $rawRoot $Split
    return @(
        Get-ChildItem $splitRoot -Directory -ErrorAction SilentlyContinue | Where-Object {
            $id = $_.Name
            $mesh = Test-Path (Join-Path $_.FullName "${id}_vh_clean_2.ply")
            if ($AnnotationsRequired) {
                $mesh -and
                (Test-Path (Join-Path $_.FullName "${id}_vh_clean_2.0.010000.segs.json")) -and
                (Test-Path (Join-Path $_.FullName "${id}.aggregation.json"))
            } else {
                $mesh
            }
        }
    ).Count
}

while ((Get-CompleteRawSceneCount "scans" $true) -lt 1513 -or (Get-CompleteRawSceneCount "scans_test" $false) -lt 100) {
    Write-Output "$logPrefix waiting for minimal ScanNet download: $(Get-CompleteRawSceneCount 'scans' $true)/1513 train+val, $(Get-CompleteRawSceneCount 'scans_test' $false)/100 test"
    Start-Sleep -Seconds 300
}

while (-not (Test-Path $python) -or -not (& $python -c "import plyfile, numpy" 2>$null)) {
    Write-Output "$logPrefix waiting for Python, NumPy, and plyfile."
    Start-Sleep -Seconds 60
}

Write-Output "$logPrefix preprocessing ScanNet."
& (Join-Path $PSScriptRoot "preprocess_scannet_litept.ps1") -Workers 8

$runDirectory = Join-Path $projectRoot "exp\scannet\semseg-litept-small-full-baseline-seed0"
$modelDirectory = Join-Path $runDirectory "model"
New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
Copy-Item -Force (Join-Path $projectRoot "artifacts\scannet_full\baseline\runtime_config.py") (Join-Path $runDirectory "config.py")
Copy-Item -Force (Join-Path $projectRoot "artifacts\scannet_full\baseline\checkpoints\model_last_cycle20.pth") (Join-Path $modelDirectory "model_last.pth")

$modules = & $python -c "import importlib.util; required=['torch','torch_geometric','spconv','plyfile','flash_attn']; print(','.join(x for x in required if importlib.util.find_spec(x) is None))"
if (-not [string]::IsNullOrWhiteSpace($modules)) {
    throw "$logPrefix preprocessing and resume layout are complete, but training cannot start; missing modules: $modules"
}

Write-Output "$logPrefix data and runtime are ready; start the project training command for cycle-21 resumption."
