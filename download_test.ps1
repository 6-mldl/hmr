Write-Host "=== Dataset Download Started ===" -ForegroundColor Green

# Create directories
$baseDir = ".\datasets"
New-Item -ItemType Directory -Force -Path $baseDir | Out-Null
New-Item -ItemType Directory -Force -Path "$baseDir\coco" | Out-Null
New-Item -ItemType Directory -Force -Path "$baseDir\mpii" | Out-Null
New-Item -ItemType Directory -Force -Path "$baseDir\up-3d" | Out-Null
New-Item -ItemType Directory -Force -Path "$baseDir\h36m" | Out-Null

Write-Host "Folders created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "datasets/"
Write-Host "  - coco/"
Write-Host "  - mpii/"
Write-Host "  - up-3d/"
Write-Host "  - h36m/"
