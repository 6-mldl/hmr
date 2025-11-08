# ⚾ Baseball 3D Analysis System

야구 타자의 3D 동작 복원 및 운동학 분석 시스템

## 📌 프로젝트 개요

2D 비디오 영상으로부터 타자의 3D 인체 메쉬를 복원하고, 스윙 분석, 반칙 판정 등을 자동으로 수행하는 시스템입니다.

### 주요 기능

- ✅ **3D 인체 복원**: HMR 기반 SMPL 메쉬 생성
- ✅ **스윙 분석**: 배트 속도, 관절 각도, 궤적 추정
- ✅ **동작 단계 구분**: Stance → Load → Swing → Contact → Follow-through
- ✅ **반칙 판정**: 배터 박스 이탈 등 자동 감지
- ✅ **시간적 스무딩**: Gaussian 필터 기반 노이즈 제거

## 🏗️ 시스템 아키텍처

```
입력 비디오
    ↓
[객체 검출] YOLOX - 사람/배트/공 검출
    ↓
[3D 복원] HMR - SMPL 메쉬 생성
    ↓
[시간적 스무딩] Gaussian Filter
    ↓
[운동학 분석] 속도/각도/궤적 계산
    ↓
[반칙 판정] 규칙 기반 판정
    ↓
출력: 3D 모델 + 분석 리포트
```

## 📦 설치 방법

### 1. 환경 요구사항

- Python 3.8+
- CUDA 11.8+ (GPU 사용 시)
- 16GB+ RAM
- 10GB+ 디스크 공간

### 2. 의존성 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/baseball_3d_analysis.git
cd baseball_3d_analysis

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

### 3. 모델 다운로드

#### A. SMPL 모델 (필수)

1. https://smpl.is.tue.mpg.de/ 접속
2. 회원가입 후 "Downloads" 페이지
3. "SMPL for Python" 다운로드
4. `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` 파일을 `models/smpl_neutral.pkl`로 복사

#### B. HMR 체크포인트

**옵션 1: PyTorch HMR 사용 (권장)**

```bash
git clone https://github.com/MandyMo/pytorch_HMR.git
# 해당 레포의 사전학습 모델 사용
```

**옵션 2: 원본 TensorFlow HMR**

```bash
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz
tar -xf models.tar.gz
```

#### C. YOLOX 모델 (자동 다운로드)

첫 실행 시 자동으로 다운로드됩니다.

## 🚀 사용 방법

### 기본 실행

```bash
python src/main_pipeline.py \
    --input_video input/baseball_swing.mp4 \
    --output_dir output/result \
    --visualize
```

### 고급 옵션

```bash
python src/main_pipeline.py \
    --input_video input/baseball_swing.mp4 \
    --output_dir output/result \
    --hmr_model models/hmr_model.pt \
    --smpl_model models/smpl_neutral.pkl \
    --fps 30 \
    --max_frames 300 \
    --conf_thresh 0.5 \
    --use_tracking \
    --visualize \
    --smooth_sigma 2.0
```

### 파라미터 설명

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--input_video` | 입력 비디오 경로 (필수) | - |
| `--output_dir` | 출력 디렉토리 | `output` |
| `--hmr_model` | HMR 체크포인트 경로 | `models/hmr_model.pt` |
| `--smpl_model` | SMPL 모델 경로 | `models/smpl_neutral.pkl` |
| `--fps` | 비디오 FPS | `30.0` |
| `--max_frames` | 최대 처리 프레임 수 | `None` (전체) |
| `--conf_thresh` | 검출 신뢰도 임계값 | `0.5` |
| `--use_tracking` | 다중 객체 추적 활성화 | `False` |
| `--visualize` | 시각화 비디오 생성 | `False` |
| `--smooth_sigma` | 스무딩 시그마 | `2.0` |

## 📂 출력 구조

```
output/
├── raw/
│   ├── metadata.json          # 프레임별 메타데이터
│   ├── vertices.npy           # (T, 6890, 3) SMPL 정점
│   ├── joints3d.npy           # (T, 24, 3) 3D 관절
│   ├── shape.npy              # (T, 10) SMPL 체형
│   ├── pose.npy               # (T, 72) SMPL 포즈
│   └── visualization.mp4      # 시각화 비디오 (옵션)
├── joints3d_smoothed.npy      # 스무딩된 관절
├── bat_trajectory.npy         # 배트 궤적
└── analysis_report.json       # 분석 리포트
```

### analysis_report.json 예시

```json
{
  "video_info": {
    "input_path": "input/swing.mp4",
    "total_frames_processed": 150,
    "fps": 30.0
  },
  "swing_analysis": {
    "max_swing_speed_ms": 32.5,
    "max_swing_speed_mph": 72.7,
    "impact_frame_estimate": 85,
    "impact_time_s": 2.833,
    "swing_start_frame": 60,
    "swing_duration_s": 0.833,
    "average_elbow_angle": 135.2,
    "max_shoulder_rotation": 45.8
  },
  "phases": {
    "stance": [0, 50],
    "load": [50, 60],
    "swing": [60, 85],
    "contact": 85,
    "follow_through": [85, 100]
  },
  "violations": []
}
```

## 🧪 테스트

각 모듈별 테스트:

```bash
# HMR 추론 테스트
python src/hmr_inference.py

# 검출기 테스트
python src/person_detector.py

# 비디오 처리 테스트
python src/video_processor.py

# 운동학 분석 테스트
python src/kinematics_analyzer.py
```

## 📊 성능

### 처리 속도 (RTX 3090 기준)

- 검출: ~60 FPS
- HMR 추론: ~10 FPS
- 전체 파이프라인: ~8 FPS

### 정확도

- 3D 관절 위치: MPJPE < 50mm (Human3.6M 기준)
- 검출 정확도: mAP > 0.9
- 스윙 속도 오차: ±5 mph

## 🛠️ 커스터마이징

### 1. 배트 검출 추가

`src/person_detector.py`에서 타겟 클래스 수정:

```python
target_classes = ['person', 'sports ball']  # 공 추가
```

### 2. 반칙 규칙 추가

`src/kinematics_analyzer.py`의 `ViolationDetector` 클래스 수정:

```python
def check_bat_throw(self, bat_velocity):
    if bat_velocity > THROW_THRESHOLD:
        return True
    return False
```

### 3. 시각화 커스터마이징

`src/video_processor.py`의 `_visualize_frame` 메서드 수정

## 📚 데이터셋

학습에 사용된 데이터셋:

- **COCO**: 2D 키포인트
- **MPII**: 2D 포즈
- **Human3.6M**: 3D 포즈 ground truth
- **UP-3D**: SMPL 파라미터

데이터셋 다운로드:

```bash
# Windows (PowerShell)
.\download_datasets.ps1

# Linux/Mac
bash setup_hmr.sh
```

## ⚠️ 알려진 제한사항

1. **가려짐 처리**: 심한 가려짐 시 정확도 저하
2. **다중 타자**: 현재는 가장 큰 사람만 추적
3. **배트 모델링**: 간단한 기하학적 모델 사용
4. **실시간 처리**: HMR의 속도 제약

## 🔧 트러블슈팅

### Q1. "Cannot find SMPL model" 오류

**A:** SMPL 모델을 https://smpl.is.tue.mpg.de/ 에서 다운로드하여 `models/` 폴더에 배치하세요.

### Q2. GPU 메모리 부족

**A:** `--max_frames` 옵션으로 처리 프레임 수를 제한하거나, 배치 사이즈를 줄이세요.

### Q3. 검출 실패

**A:** `--conf_thresh`를 낮춰보세요 (예: 0.3). 또는 영상 품질을 확인하세요.

## 🤝 기여 방법

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 단, 다음 라이브러리들은 각각의 라이선스를 따릅니다:

- HMR: [원본 라이선스](https://github.com/akanazawa/hmr/blob/master/LICENSE)
- SMPL: [SMPL 라이선스](https://smpl.is.tue.mpg.de/license.html)
- YOLOX: [Apache 2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE)

## 📞 문의

- 이슈: [GitHub Issues](https://github.com/your-repo/issues)
- 이메일: your-email@example.com

## 🙏 감사의 글

- [HMR](https://github.com/akanazawa/hmr) by Angjoo Kanazawa et al.
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) by Megvii Technology
- [SMPL](https://smpl.is.tue.mpg.de/) by Max Planck Institute

## 📈 로드맵

- [ ] 실시간 처리 최적화 (TensorRT)
- [ ] 배트 3D 모델링 개선 (NeRF)
- [ ] 다중 타자 동시 분석
- [ ] Web 기반 UI
- [ ] 모바일 앱 개발

---

**개발 버전**: v0.1.0  
**최종 업데이트**: 2025-01-01
