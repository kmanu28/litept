param(
    [string]$OutputRoot = (Join-Path $PSScriptRoot "..\data\scannet_raw")
)

$ErrorActionPreference = "Stop"
$baseUrl = "http://kaldir.vc.in.tum.de/scannet"
$trainScanIds = (Invoke-WebRequest -UseBasicParsing "$baseUrl/v2/scans.txt").Content -split "`n" | Where-Object { $_.Trim() }
$testScanIds = (Invoke-WebRequest -UseBasicParsing "$baseUrl/v2/scans_test.txt").Content -split "`n" | Where-Object { $_.Trim() }

function Get-ScanFile {
    param([string]$ScanId, [string]$Suffix, [string]$Split)
    $scanDirectory = Join-Path $OutputRoot "$Split\$ScanId"
    $destination = Join-Path $scanDirectory "$ScanId$Suffix"
    if (Test-Path -LiteralPath $destination) {
        Write-Host "Skipping existing $destination"
        return
    }
    New-Item -ItemType Directory -Force -Path $scanDirectory | Out-Null
    $url = "$baseUrl/v2/scans/$ScanId/$ScanId$Suffix"
    Write-Host "Downloading $url"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $destination
}

# These are the only raw files read by datasets/preprocessing/scannet/preprocess_scannet.py.
$trainingSuffixes = @(
    "_vh_clean_2.ply",
    "_vh_clean_2.0.010000.segs.json",
    ".aggregation.json"
)
foreach ($scanId in $trainScanIds) {
    foreach ($suffix in $trainingSuffixes) {
        Get-ScanFile -ScanId $scanId.Trim() -Suffix $suffix -Split "scans"
    }
}

# Test scenes have no public annotations; geometry is sufficient for preprocessing.
foreach ($scanId in $testScanIds) {
    Get-ScanFile -ScanId $scanId.Trim() -Suffix "_vh_clean_2.ply" -Split "scans_test"
}
