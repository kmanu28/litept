param(
    [int]$Workers = 8
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-litept\python.exe"
$rawData = Join-Path $projectRoot "data\scannet_raw"
$processedData = Join-Path $projectRoot "data\scannet_full"

if (-not (Test-Path (Join-Path $rawData "scans"))) {
    throw "Minimal ScanNet download is not present: $rawData"
}

& $python (Join-Path $projectRoot "datasets\preprocessing\scannet\preprocess_scannet.py") `
    --dataset_root $rawData `
    --output_root $processedData `
    --num_workers $Workers
