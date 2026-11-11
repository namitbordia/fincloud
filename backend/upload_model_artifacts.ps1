param([switch]$Execute)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bucket = "fincloud-market-data-unique123"
$region = "ap-south-1"
$root = Join-Path $repo "models/volatility"
$objects = @((aws s3api list-objects-v2 --bucket $bucket --prefix "models/volatility/" --region $region --output json | ConvertFrom-Json).Contents)
$existing = @{}
foreach ($item in $objects) { $existing[$item.Key] = $true }
$files = @(Get-ChildItem -LiteralPath $root -Filter "*.joblib" -File | Sort-Object Name)
if ($files.Count -ne 15) { throw "Expected all 15 trained model artifacts; found $($files.Count)." }
$records = foreach ($file in $files) {
    $key = "models/volatility/$($file.Name)"
    if ($existing.ContainsKey($key)) { throw "Refusing to overwrite existing model artifact $key" }
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $bytes = New-Object byte[] ($hash.Length / 2)
    for ($i = 0; $i -lt $bytes.Length; $i++) { $bytes[$i] = [Convert]::ToByte($hash.Substring($i * 2, 2), 16) }
    [pscustomobject]@{ Path = $file.FullName; Key = $key; Size = $file.Length; Hash = $hash; Checksum = [Convert]::ToBase64String($bytes) }
}
$mode = if ($Execute) { "EXECUTION" } else { "DRY RUN" }
Write-Output "Model artifact $mode`: $($records.Count) absent keys, $([math]::Round(($records | Measure-Object Size -Sum).Sum / 1MB, 2)) MiB; zero overwrites."
foreach ($item in $records) {
    Write-Output "s3://$bucket/$($item.Key)"
    if ($Execute) {
        aws s3 cp $item.Path "s3://$bucket/$($item.Key)" --region $region --only-show-errors --sse AES256 --checksum-algorithm SHA256
        if ($LASTEXITCODE -ne 0) { throw "Upload failed for $($item.Key)" }
        $head = aws s3api head-object --bucket $bucket --key $item.Key --region $region --checksum-mode ENABLED --output json | ConvertFrom-Json
        if ($head.ContentLength -ne $item.Size -or $head.ChecksumSHA256 -ne $item.Checksum -or $head.ServerSideEncryption -ne "AES256") { throw "Checksum/size/SSE verification failed for $($item.Key)" }
    }
}
