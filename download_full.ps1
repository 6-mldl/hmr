$ErrorActionPreference = "Stop"

Write-Host "=== Baseball 3D Analysis Dataset Download ===" -ForegroundColor Green
Write-Host ""

# Base directory
$baseDir = ".\datasets"
New-Item -ItemType Directory -Force -Path $baseDir | Out-Null

# ============================================
# 1. COCO Dataset (Large - ~19GB)
# ============================================
Write-Host "[1/4] COCO Dataset Download..." -ForegroundColor Cyan
$cocoDir = "$baseDir\coco"
New-Item -ItemType Directory -Force -Path $cocoDir | Out-Null
New-Item -ItemType Directory -Force -Path "$cocoDir\images" | Out-Null
New-Item -ItemType Directory -Force -Path "$cocoDir\annotations" | Out-Null

# COCO 2014 Train Images (13GB)
if (-Not (Test-Path "$cocoDir\images\train2014.zip")) {
    Write-Host "  - Downloading train2014.zip (13GB, may take long)..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest "http://images.cocodataset.org/zips/train2014.zip" -OutFile "$cocoDir\images\train2014.zip" -UseBasicParsing
        Write-Host "  - Extracting train2014.zip..." -ForegroundColor Yellow
        Expand-Archive "$cocoDir\images\train2014.zip" -DestinationPath "$cocoDir\images" -Force
        Write-Host "  - train2014 completed!" -ForegroundColor Green
    } catch {
        Write-Host "  X Failed to download train2014: $_" -ForegroundColor Red
    }
} else {
    Write-Host "  - train2014.zip already exists, skipping" -ForegroundColor Gray
}

# COCO 2014 Val Images (6GB)
if (-Not (Test-Path "$cocoDir\images\val2014.zip")) {
    Write-Host "  - Downloading val2014.zip (6GB)..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest "http://images.cocodataset.org/zips/val2014.zip" -OutFile "$cocoDir\images\val2014.zip" -UseBasicParsing
        Write-Host "  - Extracting val2014.zip..." -ForegroundColor Yellow
        Expand-Archive "$cocoDir\images\val2014.zip" -DestinationPath "$cocoDir\images" -Force
        Write-Host "  - val2014 completed!" -ForegroundColor Green
    } catch {
        Write-Host "  X Failed to download val2014: $_" -ForegroundColor Red
    }
} else {
    Write-Host "  - val2014.zip already exists, skipping" -ForegroundColor Gray
}

# COCO Annotations (241MB)
if (-Not (Test-Path "$cocoDir\annotations\annotations_trainval2014.zip")) {
    Write-Host "  - Downloading annotations (241MB)..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" -OutFile "$cocoDir\annotations\annotations_trainval2014.zip" -UseBasicParsing
        Write-Host "  - Extracting annotations..." -ForegroundColor Yellow
        Expand-Archive "$cocoDir\annotations\annotations_trainval2014.zip" -DestinationPath "$cocoDir" -Force
        Write-Host "  - annotations completed!" -ForegroundColor Green
    } catch {
        Write-Host "  X Failed to download annotations: $_" -ForegroundColor Red
    }
} else {
    Write-Host "  - annotations already exist, skipping" -ForegroundColor Gray
}

Write-Host "  Done: COCO Dataset" -ForegroundColor Green
Write-Host ""

# ============================================
# 2. MPII Dataset (Manual Download Required)
# ============================================
Write-Host "[2/4] MPII Dataset..." -ForegroundColor Cyan
$mpiiDir = "$baseDir\mpii"
New-Item -ItemType Directory -Force -Path $mpiiDir | Out-Null

Write-Host "  WARNING: MPII requires manual download" -ForegroundColor Yellow
Write-Host "    1. Visit: http://human-pose.mpi-inf.mpg.de/" -ForegroundColor Yellow
Write-Host "    2. Register and verify email" -ForegroundColor Yellow
Write-Host "    3. Download: mpii_human_pose_v1.tar.gz" -ForegroundColor Yellow
Write-Host "    4. Extract to: $mpiiDir" -ForegroundColor Yellow
Write-Host "  Skipping for now..." -ForegroundColor Gray
Write-Host ""

# ============================================
# 3. UP-3D Dataset
# ============================================
Write-Host "[3/4] UP-3D Dataset Download..." -ForegroundColor Cyan
$upDir = "$baseDir\up-3d"
New-Item -ItemType Directory -Force -Path $upDir | Out-Null

# UP-3D Images
if (-Not (Test-Path "$upDir\up-3d.zip")) {
    Write-Host "  - Downloading up-3d.zip..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest "https://files.is.tue.mpg.de/classner/up/up-3d.zip" -OutFile "$upDir\up-3d.zip" -UseBasicParsing
        Write-Host "  - Extracting up-3d.zip..." -ForegroundColor Yellow
        Expand-Archive "$upDir\up-3d.zip" -DestinationPath "$upDir" -Force
        Write-Host "  - up-3d images completed!" -ForegroundColor Green
    } catch {
        Write-Host "  X Failed to download up-3d: $_" -ForegroundColor Red
    }
} else {
    Write-Host "  - up-3d.zip already exists, skipping" -ForegroundColor Gray
}

# UP-3D Annotations
if (-Not (Test-Path "$upDir\up-3d-annotations.zip")) {
    Write-Host "  - Downloading up-3d-annotations.zip..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest "https://files.is.tue.mpg.de/classner/up/up-3d-annotations.zip" -OutFile "$upDir\up-3d-annotations.zip" -UseBasicParsing
        Write-Host "  - Extracting up-3d-annotations.zip..." -ForegroundColor Yellow
        Expand-Archive "$upDir\up-3d-annotations.zip" -DestinationPath "$upDir" -Force
        Write-Host "  - up-3d annotations completed!" -ForegroundColor Green
    } catch {
        Write-Host "  X Failed to download up-3d annotations: $_" -ForegroundColor Red
    }
} else {
    Write-Host "  - up-3d-annotations.zip already exists, skipping" -ForegroundColor Gray
}

Write-Host "  Done: UP-3D Dataset" -ForegroundColor Green
Write-Host ""

# ============================================
# 4. Human3.6M Dataset (Manual Download Required)
# ============================================
Write-Host "[4/4] Human3.6M Dataset..." -ForegroundColor Cyan
$h36mDir = "$baseDir\h36m"
New-Item -ItemType Directory -Force -Path $h36mDir | Out-Null

Write-Host "  WARNING: Human3.6M requires researcher approval" -ForegroundColor Yellow
Write-Host "    1. Visit: http://vision.imar.ro/human3.6m/" -ForegroundColor Yellow
Write-Host "    2. Register as researcher (1-2 days approval)" -ForegroundColor Yellow
Write-Host "    3. Download: S1~S11 subjects (100GB+)" -ForegroundColor Yellow
Write-Host "    4. Extract to: $h36mDir" -ForegroundColor Yellow
Write-Host "  Skipping for now..." -ForegroundColor Gray
Write-Host ""

# ============================================
# Summary
# ============================================
Write-Host "=== Download Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Folder Structure:"
Write-Host "datasets/"
Write-Host "  +-- coco/"
Write-Host "  |   +-- images/"
Write-Host "  |   |   +-- train2014/"
Write-Host "  |   |   +-- val2014/"
Write-Host "  |   +-- annotations/"
Write-Host "  +-- mpii/ (manual download required)"
Write-Host "  +-- up-3d/"
Write-Host "  |   +-- images/"
Write-Host "  |   +-- annotations/"
Write-Host "  +-- h36m/ (manual download required)"
Write-Host ""
Write-Host "Next Steps:"
Write-Host "  1. (Optional) Download MPII manually"
Write-Host "  2. (Optional) Apply for Human3.6M researcher access"
Write-Host "  3. Setup HMR environment and download pretrained models"
Write-Host ""
Write-Host "Note: COCO train2014 (13GB) + val2014 (6GB) = 19GB total"
Write-Host "      Download may take 1-3 hours depending on your internet speed"
