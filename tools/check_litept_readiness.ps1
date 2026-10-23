$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-litept\python.exe"
$rawRoot = Join-Path $projectRoot "data\scannet_raw"
$processedRoot = Join-Path $projectRoot "data\scannet_full"

function Write-Check([string]$Name, [bool]$Ready, [string]$Detail) {
    $state = if ($Ready) { "READY" } else { "WAITING" }
    Write-Output ("[{0}] {1}: {2}" -f $state, $Name, $Detail)
}

$requiredTrainVal = 1513
$requiredTest = 100
$rawScenes = @(Get-ChildItem (Join-Path $rawRoot "scans") -Directory -ErrorAction SilentlyContinue)
$completeRawScenes = @($rawScenes | Where-Object {
    $id = $_.Name
    (Test-Path (Join-Path $_.FullName "${id}_vh_clean_2.ply")) -and
    (Test-Path (Join-Path $_.FullName "${id}_vh_clean_2.0.010000.segs.json")) -and
    (Test-Path (Join-Path $_.FullName "${id}.aggregation.json"))
})
$rawTestScenes = @(Get-ChildItem (Join-Path $rawRoot "scans_test") -Directory -ErrorAction SilentlyContinue)
$completeRawTestScenes = @($rawTestScenes | Where-Object {
    Test-Path (Join-Path $_.FullName ("{0}_vh_clean_2.ply" -f $_.Name))
})
Write-Check "minimal ScanNet raw data" ($completeRawScenes.Count -eq $requiredTrainVal -and $completeRawTestScenes.Count -eq $requiredTest) "$($completeRawScenes.Count)/$requiredTrainVal train+val, $($completeRawTestScenes.Count)/$requiredTest test scenes"

$runtimeCheck = ""
if (Test-Path $python) {
    $runtimeCheck = & $python -c "import importlib.util; required=['torch','torch_geometric','spconv','plyfile','flash_attn','pointrope']; print(','.join(x for x in required if importlib.util.find_spec(x) is None))" 2>$null
}
$runtimeReady = (Test-Path $python) -and [string]::IsNullOrWhiteSpace($runtimeCheck)
Write-Check "LitePT runtime and extensions" $runtimeReady $(if ($runtimeReady) { "all required modules import" } elseif ($runtimeCheck) { "missing: $runtimeCheck" } else { "Python installation incomplete" })

$train = @(Get-ChildItem (Join-Path $processedRoot "train") -Directory -ErrorAction SilentlyContinue)
$val = @(Get-ChildItem (Join-Path $processedRoot "val") -Directory -ErrorAction SilentlyContinue)
$processedReady = $train.Count -eq 1201 -and $val.Count -eq 312
Write-Check "processed ScanNet" $processedReady "$($train.Count)/1201 train, $($val.Count)/312 validation scenes"

$checkpoint = Join-Path $projectRoot "artifacts\scannet_full\baseline\checkpoints\model_last_cycle20.pth"
Write-Check "resume checkpoint" (Test-Path $checkpoint) $checkpoint
