#!/bin/bash
# HMR 환경 설정 스크립트 (Linux/Mac)
# 실행: bash setup_hmr.sh

set -e

echo "=== HMR 환경 설정 시작 ==="

# 1. PyTorch HMR 클론 (Python 3 버전)
echo "[1/5] PyTorch HMR 레포 클론..."
if [ ! -d "pytorch_HMR" ]; then
    git clone https://github.com/MandyMo/pytorch_HMR.git
    cd pytorch_HMR
else
    cd pytorch_HMR
    git pull
fi

# 2. 가상환경 생성
echo "[2/5] 가상환경 생성..."
python3 -m venv venv_hmr
source venv_hmr/bin/activate

# 3. 필요 패키지 설치
echo "[3/5] 패키지 설치..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib scipy scikit-image
pip install neural-renderer-pytorch
pip install trimesh pyrender

# 4. SMPL 모델 다운로드 준비
echo "[4/5] SMPL 모델 다운로드 안내..."
mkdir -p models
echo ""
echo "⚠️  SMPL 모델 수동 다운로드 필요:"
echo "   1. https://smpl.is.tue.mpg.de/ 접속"
echo "   2. 회원가입 후 'Download' 페이지 이동"
echo "   3. 'SMPL for Python' 다운로드 (SMPL_python_v.1.0.0.zip)"
echo "   4. 압축 해제 후 models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl 복사"
echo ""

# 5. 사전학습 모델 다운로드
echo "[5/5] 사전학습 모델 다운로드..."
cd models
if [ ! -f "hmr_model.pt" ]; then
    # PyTorch 버전 모델 (있다면)
    echo "사전학습 모델은 원본 레포에서 제공하는 것을 사용하세요."
    echo "또는 직접 학습하거나 다른 소스에서 다운로드 필요"
fi
cd ..

echo ""
echo "=== 환경 설정 완료 ==="
echo ""
echo "다음 단계:"
echo "  1. SMPL 모델 수동 다운로드 및 설치"
echo "  2. 테스트: python demo.py --img test_image.jpg"
