$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-litept\python.exe"
$logDirectory = Join-Path $projectRoot ".tools"
$cacheDirectory = Join-Path $projectRoot ".pip-cache"
$tempDirectory = Join-Path $projectRoot ".pip-tmp"
$env:PIP_CACHE_DIR = $cacheDirectory
$env:TEMP = $tempDirectory
$env:TMP = $tempDirectory

while (-not (Test-Path (Join-Path $projectRoot ".venv-litept\Lib\site-packages\torch"))) {
    Start-Sleep -Seconds 30
}

& $python -m pip install --upgrade pip
& $python -m pip install h5py pyyaml sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm colorhash torchmetrics
& $python -m pip install torch-cluster torch-scatter torch-sparse torch-geometric==2.5.3 -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
& $python -m pip install spconv-cu124
