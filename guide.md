HMR 실행 전체 가이드 (Vast.ai + GitHub + Google Drive)
버전: 최종 통합 실무용 (2025.11 기준)
________________________________________
0️⃣ 구성 개요
코드 저장소(GitHub) → hmr_inference.py, main_pipeline.py, person_detector.py, video_processor.py, kinematics_analyzer.py, setup_hmr.sh, README.md, QUICK_START.md, SUMMARY.md
대용량 리소스(Google Drive 공유 폴더) → 폴더 링크: https://drive.google.com/drive/folders/1BqNIK4wR0aCwrbymVJXJaaXng7OWMP4-
⚙️ 원칙
•	GitHub = 코드, 문서, 스크립트
•	Google Drive = 대용량 데이터셋, 모델, SMPL 리소스
•	공유 폴더를 통째로 다운로드하여 사용
________________________________________
1️⃣ Vast.ai 인스턴스 생성
1.	https://vast.ai/console/create/ 접속
2.	설정값:
o	GPU: RTX 3090 / 4090 / A5000 이상 (VRAM ≥ 12GB)
o	Disk: 최소 120GB (권장 150GB)
o	Image: pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
o	Applications: Jupyter, SSH
o	OS: Ubuntu 22.04
3.	SSH 키 등록 (Windows PowerShell):
4.	ssh-keygen -t ed25519 -C "vast-ai" -f "$env:USERPROFILE\.ssh\id_ed25519"
5.	Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
→ Vast.ai → Account → SSH Keys → Add Key → 인스턴스 Stop 후 Start (재시작 필수)
6.	SSH 연결:
7.	ssh -p <PORT> -i "$env:USERPROFILE\.ssh\id_ed25519" `
8.	root@<VAST_IP> -L 8080:localhost:8080
________________________________________
2️⃣ SSH 접속 후 기본 환경 구성
mkdir -p /workspace/HMR_Project
cd /workspace/HMR_Project
________________________________________
3️⃣ GitHub 코드 받기
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>
예상 구조:
/workspace/HMR_Project/<YOUR_REPO>/
  hmr_inference.py
  main_pipeline.py
  person_detector.py
  video_processor.py
  kinematics_analyzer.py
  setup_hmr.sh
  README.md
________________________________________
4️⃣ Google Drive 데이터 받기
✅ 방법: gdown으로 공유 폴더 다운로드
1.	gdown 설치
2.	pip install gdown
3.	공유 폴더 전체 다운로드
4.	cd /workspace/HMR_Project
5.	
6.	# 공유 폴더 전체를 재귀적으로 다운로드
7.	gdown --folder https://drive.google.com/drive/folders/1BqNIK4wR0aCwrbymVJXJaaXng7OWMP4- \
8.	  --output ./hmr_data --remaining-ok
9.	다운로드 확인
10.	ls -lh ./hmr_data
11.	데이터를 적절한 위치로 이동 및 압축 해제
cd /workspace/HMR_Project/<YOUR_REPO>

# -----------------------------
# 1️⃣ 디렉토리 생성
# -----------------------------
mkdir -p datasets models smpl

# -----------------------------
# 2️⃣ COCO 데이터셋 (zip 파일)
# -----------------------------
if [ -f ../hmr_data/train2017.zip ]; then
  echo "[INFO] Extracting train2017.zip ..."
  unzip -q ../hmr_data/train2017.zip -d datasets/
fi

if [ -f ../hmr_data/val2017.zip ]; then
  echo "[INFO] Extracting val2017.zip ..."
  unzip -q ../hmr_data/val2017.zip -d datasets/
fi

if [ -f ../hmr_data/annotations_trainval2017.zip ]; then
  echo "[INFO] Extracting annotations_trainval2017.zip ..."
  unzip -q ../hmr_data/annotations_trainval2017.zip -d datasets/
fi

# -----------------------------
# 3️⃣ 모델 파일 (tar.gz)
# -----------------------------
if [ -f ../hmr_data/models.tar.gz ]; then
  echo "[INFO] Extracting models.tar.gz ..."
  tar -xzf ../hmr_data/models.tar.gz -C models/
fi

# -----------------------------
# 4️⃣ SMPL 파일 (zip)
# -----------------------------
if [ -f ../hmr_data/SMPL_python_v.1.1.0.zip ]; then
  echo "[INFO] Extracting SMPL_python_v.1.1.0.zip ..."
  unzip -q ../hmr_data/SMPL_python_v.1.1.0.zip -d smpl/
fi

# -----------------------------
# 5️⃣ up-3d.zip (새로 추가된 부분)
# -----------------------------
if [ -f ../hmr_data/up-3d.zip ]; then
  echo "[INFO] Extracting up-3d.zip ..."
  unzip -q ../hmr_data/up-3d.zip -d datasets/
fi

# -----------------------------
# 6️⃣ 폴더 형태로 이미 존재하는 경우
# -----------------------------
if [ -d ../hmr_data/up-3d ]; then
  echo "[INFO] Copying up-3d folder ..."
  cp -r ../hmr_data/up-3d datasets/
fi

if [ -d ../hmr_data/SMPL_python ]; then
  echo "[INFO] Copying SMPL_python folder ..."
  cp -r ../hmr_data/SMPL_python smpl/
fi

if [ -d ../hmr_data/models ]; then
  echo "[INFO] Copying models folder ..."
  cp -r ../hmr_data/models/* models/
fi

# -----------------------------
# 7️⃣ 중복 폴더 정리 (up-3d/up-3d 구조 방지)
# -----------------------------
if [ -d datasets/up-3d/up-3d ]; then
  echo "[INFO] Flattening nested up-3d folder ..."
  mv datasets/up-3d/up-3d/* datasets/up-3d/
  rm -rf datasets/up-3d/up-3d
fi

echo "[SUCCESS] Dataset setup complete!"________________________________________
5️⃣ 데이터 구조 점검
cd /workspace/HMR_Project/<YOUR_REPO>
tree -L 2 -d
정상 구조 예시:
.
├── datasets
│   ├── train2017
│   ├── val2017
│   ├── annotations
│   └── up-3d
├── models
│   └── (model.ckpt-667589.* files)
└── smpl
    └── SMPL_python_v.1.1.0
________________________________________
6️⃣ 환경 세팅 (자동 설치)
setup_hmr.sh 내용 확인/수정:
#!/bin/bash

# 시스템 패키지 설치
apt-get update -y
apt-get install -y ffmpeg libegl1-mesa libgbm1

# Python 패키지 설치
pip install --upgrade pip
pip install numpy scipy opencv-python matplotlib tqdm trimesh pyrender pillow smplx
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install gdown

# 환경 변수 설정
export PYOPENGL_PLATFORM=egl

echo "✅ Setup complete!"
실행:
chmod +x setup_hmr.sh
./setup_hmr.sh
________________________________________
7️⃣ 스모크 테스트 (단일 이미지)
export PYOPENGL_PLATFORM=egl

python hmr_inference.py \
  --model_path ./models/model.ckpt-667589 \
  --smpl_path ./models/neutral_smpl_with_cocoplus_reg.pkl \
  --img_path ./datasets/val2017/000000000139.jpg
확인 사항:
•	✅ GPU 인식됨
•	✅ 모델 로드 성공
•	✅ 3D 메시 생성됨
•	✅ 출력 파일 저장됨
________________________________________
8️⃣ 전체 파이프라인 실행
🎞 단일 비디오 처리
export PYOPENGL_PLATFORM=egl

python main_pipeline.py \
  --input_video "./datasets/test_video.mp4" \
  --output_dir "./output/run1" \
  --fps 30 \
  --visualize \
  --use_tracking
📂 폴더 내 비디오 일괄 처리
export PYOPENGL_PLATFORM=egl

for video in ./datasets/videos/*.mp4; do
  name=$(basename "$video" .mp4)
  python main_pipeline.py \
    --input_video "$video" \
    --output_dir "./output/$name" \
    --fps 30 \
    --visualize \
    --use_tracking
done
________________________________________
9️⃣ 결과 회수 (Vast → 로컬)
Windows PowerShell:
scp -r -P <PORT> -i "$env:USERPROFILE\.ssh\id_ed25519" `
  "root@<VAST_IP>:/workspace/HMR_Project/<YOUR_REPO>/output" `
  ".\output_from_vast"
Mac/Linux:
scp -r -P <PORT> -i ~/.ssh/id_ed25519 \
  root@<VAST_IP>:/workspace/HMR_Project/<YOUR_REPO>/output \
  ./output_from_vast
________________________________________
🔍 10️⃣ 문제 해결
문제	원인	해결
Permission denied (publickey)	SSH 키 미등록	Vast.ai에 키 등록 후 인스턴스 재시작
gdown 폴더 다운로드 실패	권한 문제	--remaining-ok 옵션 사용
EGL / OpenGL 에러	렌더링 라이브러리 누락	apt install libgbm1 libegl1-mesa + export PYOPENGL_PLATFORM=egl
경로 인식 실패	중첩된 폴더 구조	mv datasets/train2017/train2017/* datasets/train2017/
CUDA out of memory	배치 크기 과다	--batch_size 1 또는 더 작은 GPU 사용
________________________________________
✅ 11️⃣ 최종 폴더 구조
/workspace/HMR_Project/<YOUR_REPO>/
├── datasets/
│   ├── train2017/
│   ├── val2017/
│   ├── annotations/
│   └── up-3d/
├── models/
│   ├── model.ckpt-667589.data-00000-of-00001
│   ├── model.ckpt-667589.index
│   ├── model.ckpt-667589.meta
│   └── neutral_smpl_with_cocoplus_reg.pkl
├── smpl/
│   └── SMPL_python_v.1.1.0/
├── hmr_inference.py
├── main_pipeline.py
├── person_detector.py
├── video_processor.py
├── kinematics_analyzer.py
├── setup_hmr.sh
└── output/
________________________________________
📦 한 줄 요약
"GitHub에서 코드 클론 → gdown으로 공유 폴더 다운로드 → setup_hmr.sh 실행 → hmr_inference.py 테스트 → main_pipeline.py 실행"
________________________________________
🚀 빠른 시작 (복붙용 전체 명령어)
# 1. 프로젝트 폴더 생성
mkdir -p /workspace/HMR_Project && cd /workspace/HMR_Project

# 2. GitHub 코드 받기
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>

# 3. Google Drive 데이터 받기
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1BqNIK4wR0aCwrbymVJXJaaXng7OWMP4- \
  --output ../hmr_data --remaining-ok

# 4. 환경 설정
chmod +x setup_hmr.sh && ./setup_hmr.sh

# 5. 데이터 배치 (압축 해제)
mkdir -p datasets models smpl
cd ../hmr_data && for f in *.zip; do unzip -q "$f" -d ../$(basename "$f" .zip); done
cd ../<REPO>

# 6. 테스트 실행
export PYOPENGL_PLATFORM=egl
python hmr_inference.py --model_path ./models/model.ckpt-667589 \
  --smpl_path ./models/neutral_smpl_with_cocoplus_reg.pkl \
  --img_path ./datasets/val2017/000000000139.jpg
________________________________________
이제 팀원은 이 가이드만 보고 처음부터 끝까지 실행할 수 있어! 🎯

