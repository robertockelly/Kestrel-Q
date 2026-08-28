# Copyright 2026 Kestrel-Q contributors
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$Revision,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string]$RepositoryId = 'Qwen/Qwen3.8-Flash-Next',

    [string]$CacheRoot = '.research-cache/model-baseline',

    [ValidatePattern('^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')]
    [string]$CapturedAtUtc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Task 1.0 deliberately names every file it may fetch. Adding an upstream file
# requires a source change and review; repository-wide or pattern downloads are
# never performed by this tool.
$downloadAllowlist = @(
    '.gitattributes',
    'LICENSE',
    'README.md',
    'chat_template.jinja',
    'config.json',
    'generation_config.json',
    'merges.txt',
    'model.safetensors.index.json',
    'preprocessor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'video_preprocessor_config.json',
    'vocab.json'
)

function Get-Classification {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ($Path -match '^model-\d{5}-of-\d{5}\.safetensors$') { return 'weight_shard' }
    if ($Path -eq 'model.safetensors.index.json') { return 'checkpoint_index' }
    if ($Path -in @('config.json', 'generation_config.json')) { return 'configuration' }
    if ($Path -in @('chat_template.jinja', 'merges.txt', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json')) { return 'tokenizer' }
    if ($Path -in @('preprocessor_config.json', 'video_preprocessor_config.json')) { return 'processor' }
    if ($Path -eq 'LICENSE') { return 'license' }
    if ($Path -eq 'README.md') { return 'documentation' }
    if ($Path -eq '.gitattributes') { return 'repository_metadata' }
    return 'other'
}

foreach ($allowedPath in $downloadAllowlist) {
    if ($allowedPath -match '(?i)\.safetensors$') {
        throw "Unsafe allowlist entry rejected: $allowedPath"
    }
}

$repositorySegments = $RepositoryId -split '/'
if ($repositorySegments.Count -ne 2) {
    throw 'RepositoryId must have exactly one organization/name separator.'
}

$encodedRepository = ($repositorySegments | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
$apiUri = "https://huggingface.co/api/models/$encodedRepository/revision/$Revision`?blobs=true"
$model = Invoke-RestMethod -Method Get -Uri $apiUri -Headers @{ 'User-Agent' = 'Kestrel-Q-Task-1.0' }

if ($model.sha -ne $Revision) {
    throw "Pinned revision mismatch: requested $Revision, API returned $($model.sha)."
}

$siblingsByPath = @{}
foreach ($sibling in $model.siblings) {
    $siblingsByPath[$sibling.rfilename] = $sibling
}

foreach ($allowedPath in $downloadAllowlist) {
    if (-not $siblingsByPath.ContainsKey($allowedPath)) {
        throw "Allowlisted file is absent from pinned revision: $allowedPath"
    }
}

$cacheDirectory = Join-Path $CacheRoot $Revision
New-Item -ItemType Directory -Force -Path $cacheDirectory | Out-Null

$downloadEvidence = @{}
foreach ($allowedPath in $downloadAllowlist) {
    $sibling = $siblingsByPath[$allowedPath]
    $encodedPath = (($allowedPath -split '/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
    $downloadUri = "https://huggingface.co/$encodedRepository/resolve/$Revision/$encodedPath`?download=true"
    $localPath = Join-Path $cacheDirectory $allowedPath
    $localParent = Split-Path -Parent $localPath
    New-Item -ItemType Directory -Force -Path $localParent | Out-Null

    Invoke-WebRequest -Method Get -Uri $downloadUri -OutFile $localPath -Headers @{ 'User-Agent' = 'Kestrel-Q-Task-1.0' }
    $localFile = Get-Item -LiteralPath $localPath
    if ([int64]$localFile.Length -ne [int64]$sibling.size) {
        throw "Size mismatch for ${allowedPath}: expected $($sibling.size), received $($localFile.Length)."
    }

    $sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $localPath).Hash.ToLowerInvariant()
    $lfsProperty = $sibling.PSObject.Properties['lfs']
    $lfs = if ($null -ne $lfsProperty) { $lfsProperty.Value } else { $null }
    if ($null -ne $lfs -and $null -ne $lfs.sha256 -and $sha256 -ne $lfs.sha256) {
        throw "LFS SHA-256 mismatch for $allowedPath."
    }

    $downloadEvidence[$allowedPath] = [ordered]@{
        cache_relative_path = (Join-Path $Revision $allowedPath).Replace('\', '/')
        sha256 = $sha256
    }
}

$files = @()
foreach ($sibling in ($model.siblings | Sort-Object rfilename)) {
    $path = [string]$sibling.rfilename
    $isWeightShard = $path -match '^model-\d{5}-of-\d{5}\.safetensors$'
    $downloaded = $downloadEvidence.ContainsKey($path)
    $lfsSha256 = $null
    $lfsSize = $null
    $lfsProperty = $sibling.PSObject.Properties['lfs']
    $lfs = if ($null -ne $lfsProperty) { $lfsProperty.Value } else { $null }
    if ($null -ne $lfs) {
        $lfsSha256 = $lfs.sha256
        $lfsSize = $lfs.size
    }

    $entry = [pscustomobject][ordered]@{
        path = $path
        size_bytes = [int64]$sibling.size
        classification = Get-Classification -Path $path
        is_weight_shard = $isWeightShard
        downloaded = $downloaded
        upstream_blob_id = $sibling.blobId
        upstream_lfs_sha256 = $lfsSha256
        upstream_lfs_size_bytes = $lfsSize
        downloaded_sha256 = if ($downloaded) { $downloadEvidence[$path].sha256 } else { $null }
    }
    $files += $entry
}

$weightShards = @($files | Where-Object { $_.is_weight_shard })
$downloadedFiles = @($files | Where-Object { $_.downloaded })
if (@($downloadedFiles | Where-Object { $_.is_weight_shard }).Count -ne 0) {
    throw 'Safety invariant failed: a weight shard was downloaded.'
}

$manifest = [ordered]@{
    schema_version = 1
    generated_by = 'tools/capture-model-baseline.ps1'
    repository_id = $RepositoryId
    repository_url = "https://huggingface.co/$RepositoryId"
    revision = $Revision
    captured_at_utc = $CapturedAtUtc
    upstream_last_modified = ([DateTimeOffset]$model.lastModified).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    upstream_reported_storage_bytes = [int64]$model.usedStorage
    file_count = $files.Count
    weight_shard_count = $weightShards.Count
    weight_shard_bytes = [int64](($weightShards | Measure-Object -Property size_bytes -Sum).Sum)
    downloaded_file_count = $downloadedFiles.Count
    download_allowlist = $downloadAllowlist
    safety = [ordered]@{
        unrestricted_download_supported = $false
        safetensors_download_forbidden = $true
        downloaded_weight_shard_count = 0
    }
    files = $files
}

$outputParent = Split-Path -Parent $OutputPath
if ($outputParent) {
    New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
}
$json = $manifest | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText((Join-Path (Get-Location) $OutputPath), $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))

[pscustomobject]@{
    Repository = $RepositoryId
    Revision = $Revision
    Files = $files.Count
    WeightShardsListed = $weightShards.Count
    MetadataFilesDownloaded = $downloadedFiles.Count
    WeightShardsDownloaded = 0
    Output = $OutputPath
}
