param(
    [Parameter(Mandatory=$true)][string]$Destination,
    [string]$GcloudCommand = 'gcloud'
)
$ErrorActionPreference = 'Stop'
# Requires an already authenticated account with legitimate Waymo access.
# This script never signs in, accepts terms, creates a cloud project or enables billing.
$targetRoot = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null
$bucketRoot = 'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario'
foreach ($splitSpec in @(@('training',12,1000), @('validation',10,150))) {
    $splitName = [string]$splitSpec[0]
    $splitCount = [int]$splitSpec[1]
    $totalShards = [int]$splitSpec[2]
    $splitPath = Join-Path $targetRoot $splitName
    New-Item -ItemType Directory -Force -Path $splitPath | Out-Null
    $sourceObjects = @()
    for ($shardIndex = 0; $shardIndex -lt $splitCount; $shardIndex++) {
        $fileName = '{0}.tfrecord-{1:D5}-of-{2:D5}' -f $splitName,$shardIndex,$totalShards
        $sourceObjects += "$bucketRoot/$splitName/$fileName"
    }
    $manifestPath = Join-Path $targetRoot "$splitName-download.csv"
    & $GcloudCommand storage cp --no-clobber "--manifest-path=$manifestPath" @sourceObjects $splitPath
    if ($LASTEXITCODE -ne 0) { throw "Official $splitName download did not complete; no experiment was started." }
}
Write-Output "Fixed official shards downloaded to $targetRoot. Run preparation to verify TFRecord integrity."
