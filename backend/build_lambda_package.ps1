$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputDir = Join-Path $PSScriptRoot "dist"
$outputFile = Join-Path $outputDir "fincloud-api.zip"
$image = "fincloud-lambda-package:local"
$container = "fincloud-lambda-package-$([guid]::NewGuid().ToString('N'))"

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
docker build --platform linux/amd64 -f (Join-Path $PSScriptRoot "Dockerfile.lambda-package") -t $image $repo
if ($LASTEXITCODE -ne 0) { throw "Docker build failed." }
docker create --name $container $image | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Could not create package container." }
try {
    docker cp "${container}:/tmp/fincloud-api.zip" $outputFile
    if ($LASTEXITCODE -ne 0) { throw "Could not extract the Lambda zip." }
}
finally {
    docker rm $container | Out-Null
}

& (Join-Path $repo ".venv\Scripts\python.exe") -c "import os,sys,zipfile; p=r'$outputFile'; z=zipfile.ZipFile(p); n=sum(i.file_size for i in z.infolist()); print('compressed='+str(os.path.getsize(p))+' uncompressed='+str(n)); sys.exit(0 if n < 250*1024*1024 else 1)"
if ($LASTEXITCODE -ne 0) { throw "Lambda package exceeds the 250 MiB unzipped limit or failed validation." }
