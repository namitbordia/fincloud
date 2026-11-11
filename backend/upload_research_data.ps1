param([switch]$Execute)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bucket = "fincloud-market-data-unique123"
$region = "ap-south-1"
$mapping = [System.Collections.Generic.List[object]]::new()

function Get-RelativePath([string]$Base, [string]$Target) {
    $basePath = (Resolve-Path -LiteralPath $Base).Path.TrimEnd('\') + '\'
    $targetPath = (Resolve-Path -LiteralPath $Target).Path
    $baseUri = [Uri]::new($basePath)
    $targetUri = [Uri]::new($targetPath)
    return [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('\', '/')
}

function Add-Files([string]$Directory, [string]$Prefix, [string[]]$Names = @()) {
    $root = Join-Path $repo $Directory
    if ($Names.Count -gt 0) {
        foreach ($name in $Names) {
            $path = Join-Path $root $name
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing expected source file: $path" }
            $mapping.Add([pscustomobject]@{ Path = $path; Key = "$Prefix/$name" })
        }
        return
    }
    foreach ($file in Get-ChildItem -LiteralPath $root -File -Recurse | Where-Object { $_.Extension -in @('.csv', '.parquet', '.json') }) {
        $relative = Get-RelativePath $root $file.FullName
        $mapping.Add([pscustomobject]@{ Path = $file.FullName; Key = "$Prefix/$relative" })
    }
}

Add-Files "data/raw/nse" "bronze"
Add-Files "data/processed/silver" "silver"
Add-Files "data/processed/analytics" "gold/analytics"
Add-Files "models/volatility" "models/volatility"
Add-Files "data/processed/forecasting/ml" "experiments/ml" @('feature_importance.csv','ml_forecasts.csv','ml_manifest.json','ml_model_metrics.csv','ml_stock_metrics.csv','ml_validation_report.json','model_comparison.csv','fincloud_research_summary.json')
Add-Files "data/processed/forecasting/statistical" "experiments/statistical" @('statistical_forecasts.csv','statistical_manifest.json','statistical_model_metrics.csv','statistical_stock_metrics.csv','statistical_validation_report.json')
Add-Files "data/processed/optimization" "experiments/optimization" @('backtest_validation_report.json','optimization_diagnostics.json','optimization_manifest.json','optimized_weights.csv','strategy_performance.csv','strategy_risk_metrics.csv')

$duplicates = $mapping | Group-Object Key | Where-Object Count -gt 1
if ($duplicates) { throw "Duplicate target keys: $($duplicates.Name -join ', ')" }
$objects = @((aws s3api list-objects-v2 --bucket $bucket --region $region --output json | ConvertFrom-Json).Contents)
$existing = @{}
foreach ($item in $objects) { $existing[$item.Key] = $true }
$conflicts = @($mapping | Where-Object { $existing.ContainsKey($_.Key) })
if ($conflicts.Count -gt 0) { throw "Refusing to overwrite existing objects: $($conflicts.Key -join ', ')" }

$inventory = foreach ($item in $mapping) {
    $hash = (Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    $bytes = New-Object byte[] ($hash.Length / 2)
    for ($i = 0; $i -lt $bytes.Length; $i++) { $bytes[$i] = [Convert]::ToByte($hash.Substring($i * 2, 2), 16) }
    [pscustomobject]@{ key = $item.Key; size_bytes = (Get-Item -LiteralPath $item.Path).Length; sha256 = $hash; checksum_sha256 = [Convert]::ToBase64String($bytes); source = Get-RelativePath $repo $item.Path }
}
$manifestPath = Join-Path $repo "data/processed/s3_manifests/fincloud_live_deployment_manifest.json"
$manifest = [ordered]@{ schema_version = 1; status = if ($Execute) { "uploaded" } else { "dry-run" }; region = $region; bucket = $bucket; data_source = "Yahoo Finance through yfinance; supplementary research data, not official NSE-certified data"; generated_at_utc = [DateTime]::UtcNow.ToString("o"); object_count = $inventory.Count; objects = @($inventory) }
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8
$metaKey = "metadata/fincloud_live_deployment_manifest.json"
if ($existing.ContainsKey($metaKey)) { throw "Refusing to overwrite existing manifest object $metaKey" }

$mode = if ($Execute) { "EXECUTION" } else { "DRY RUN" }
Write-Output "S3 data upload $mode`: $($mapping.Count) new data objects, $([math]::Round(($inventory | Measure-Object size_bytes -Sum).Sum / 1MB, 2)) MiB. Existing target conflicts: 0."
foreach ($item in $mapping) {
    Write-Output "s3://$bucket/$($item.Key) <- $(Get-RelativePath $repo $item.Path)"
    if ($Execute) {
        aws s3 cp $item.Path "s3://$bucket/$($item.Key)" --region $region --only-show-errors --sse AES256 --checksum-algorithm SHA256
        if ($LASTEXITCODE -ne 0) { throw "Upload failed for $($item.Key)" }
        $head = aws s3api head-object --bucket $bucket --key $item.Key --region $region --checksum-mode ENABLED --output json | ConvertFrom-Json
        $expected = $inventory | Where-Object key -eq $item.Key | Select-Object -First 1
        if ($head.ContentLength -ne $expected.size_bytes -or $head.ChecksumSHA256 -ne $expected.checksum_sha256 -or $head.ServerSideEncryption -ne "AES256") { throw "Size, SHA-256, or SSE-S3 verification failed for $($item.Key)" }
    }
}

if ($Execute) {
    $manifest.status = "uploaded"
    $manifestBytes = [System.Text.Encoding]::UTF8.GetBytes(($manifest | ConvertTo-Json -Depth 8))
    $tmp = Join-Path $env:TEMP "fincloud-live-manifest.json"
    [System.IO.File]::WriteAllBytes($tmp, $manifestBytes)
    aws s3 cp $tmp "s3://$bucket/$metaKey" --region $region --only-show-errors --sse AES256 --checksum-algorithm SHA256
    if ($LASTEXITCODE -ne 0) { throw "Manifest upload failed." }
    $head = aws s3api head-object --bucket $bucket --key $metaKey --region $region --checksum-mode ENABLED --output json | ConvertFrom-Json
    if ($head.ContentLength -ne $manifestBytes.Length -or $head.ServerSideEncryption -ne "AES256") { throw "Deployment manifest verification failed." }
    Write-Output "Verified deployment manifest: s3://$bucket/$metaKey"
}
