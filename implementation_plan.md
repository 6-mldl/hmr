# 야구 타자 3D 복원 프로젝트 구현 계획서

## 📊 전체 구현 로드맵

### Phase 0: 환경 준비 (1주)
- [x] 데이터셋 다운로드 스크립트 작성
- [x] HMR 환경 설정 스크립트 작성
- [ ] 사용자 실행: 데이터셋 다운로드
- [ ] 사용자 실행: SMPL 모델 수동 다운로드
- [ ] 사용자 실행: MPII, H36M 연구자 신청

### Phase 1: 기본 추론 파이프라인 (1주)
- [ ] HMR 단일 이미지 추론 테스트
- [ ] 사람 검출기 연동 (YOLOX)
- [ ] 프레임 단위 비디오 처리
- [ ] 3D 시각화 (Open3D/PyRender)

### Phase 2: 야구 특화 기능 (2주)
- [ ] 배트 검출 및 추적
- [ ] 시간적 스무딩 (Temporal Smoothing)
- [ ] 배트 3D 모델링
- [ ] 관절 각도/속도 계산

### Phase 3: 분석 및 판정 (1주)
- [ ] 스윙 분석 로직
- [ ] 반칙 판정 규칙
- [ ] 결과 리포트 생성

---

## 🔧 구현 상세 계획

### Module 1: HMR 기본 추론 (즉시 제공 가능)

#### 1-1. 단일 이미지 추론 래퍼
```python
import torch
import cv2
import numpy as np
from models import hmr

class HMRInference:
    def __init__(self, model_path, smpl_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = hmr.HMR().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def preprocess(self, img_path, bbox):
        """이미지 전처리 및 크롭"""
        img = cv2.imread(img_path)
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (224, 224))
        crop = crop.astype(np.float32) / 255.0
        crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
        return crop.to(self.device)
    
    def predict(self, img_path, bbox):
        """3D 포즈 예측"""
        img_tensor = self.preprocess(img_path, bbox)
        
        with torch.no_grad():
            pred = self.model(img_tensor)
            
        return {
            'vertices': pred['vertices'][0].cpu().numpy(),  # (6890, 3)
            'joints': pred['joints3d'][0].cpu().numpy(),     # (24, 3)
            'shape': pred['shape'][0].cpu().numpy(),         # (10,)
            'pose': pred['pose'][0].cpu().numpy()            # (72,)
        }

# 사용 예시
hmr_model = HMRInference('models/hmr_model.pt', 'models/smpl_neutral.pkl')
result = hmr_model.predict('image.jpg', bbox=[100, 50, 300, 500])
```

#### 1-2. 비디오 처리 파이프라인
```python
import cv2
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, hmr_model, detector):
        self.hmr = hmr_model
        self.detector = detector
        
    def process_video(self, video_path, output_path):
        """비디오 프레임별 처리"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = []
        
        for frame_idx in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. 사람 검출
            detections = self.detector.detect(frame)
            
            # 2. 각 사람에 대해 HMR 실행
            for det in detections:
                if det['class'] == 'person':
                    bbox = det['bbox']
                    # 임시 저장
                    cv2.imwrite(f'temp_frame_{frame_idx}.jpg', frame)
                    
                    # HMR 추론
                    pred = self.hmr.predict(f'temp_frame_{frame_idx}.jpg', bbox)
                    pred['frame_idx'] = frame_idx
                    pred['bbox'] = bbox
                    results.append(pred)
                    
        cap.release()
        
        # 결과 저장
        np.save(output_path, results)
        return results
```

---

### Module 2: 사람 검출기 연동 (즉시 제공 가능)

#### 2-1. YOLOX 래퍼
```python
import torch

class PersonDetector:
    def __init__(self, model_name='yolox-x'):
        self.model = torch.hub.load('Megvii-BaseDetection/YOLOX', model_name)
        self.model.eval()
        
    def detect(self, frame):
        """사람 검출 (bbox 반환)"""
        outputs = self.model(frame)
        
        detections = []
        for output in outputs:
            if output is None:
                continue
                
            bboxes = output[:, :4]  # x1, y1, x2, y2
            scores = output[:, 4]
            classes = output[:, 6]
            
            # 사람 클래스만 필터링 (COCO class 0)
            person_mask = classes == 0
            
            for bbox, score in zip(bboxes[person_mask], scores[person_mask]):
                if score > 0.5:
                    detections.append({
                        'class': 'person',
                        'bbox': bbox.cpu().numpy().astype(int).tolist(),
                        'confidence': float(score)
                    })
                    
        return detections

# 사용 예시
detector = PersonDetector()
detections = detector.detect(frame)
```

---

### Module 3: 3D 시각화 (즉시 제공 가능)

#### 3-1. Open3D 기반 렌더링
```python
import open3d as o3d
import numpy as np

class MeshVisualizer:
    def __init__(self, smpl_faces):
        self.faces = smpl_faces
        
    def create_mesh(self, vertices):
        """SMPL vertices를 Open3D mesh로 변환"""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        mesh.compute_vertex_normals()
        
        # 색상 추가
        mesh.paint_uniform_color([0.7, 0.7, 0.9])
        return mesh
    
    def visualize(self, vertices):
        """3D 메쉬 시각화"""
        mesh = self.create_mesh(vertices)
        o3d.visualization.draw_geometries([mesh])
        
    def save_mesh(self, vertices, output_path):
        """OBJ 파일로 저장"""
        mesh = self.create_mesh(vertices)
        o3d.io.write_triangle_mesh(output_path, mesh)

# 사용 예시
visualizer = MeshVisualizer(smpl_faces)
visualizer.visualize(result['vertices'])
visualizer.save_mesh(result['vertices'], 'output.obj')
```

---

### Module 4: 시간적 스무딩 (즉시 제공 가능)

#### 4-1. 1D Gaussian Filter
```python
from scipy.ndimage import gaussian_filter1d

class TemporalSmoother:
    def __init__(self, sigma=2.0):
        self.sigma = sigma
        
    def smooth_sequence(self, joints_sequence):
        """
        joints_sequence: (T, 24, 3) - T프레임, 24관절, xyz
        """
        T, J, D = joints_sequence.shape
        smoothed = np.zeros_like(joints_sequence)
        
        for j in range(J):
            for d in range(D):
                smoothed[:, j, d] = gaussian_filter1d(
                    joints_sequence[:, j, d], 
                    sigma=self.sigma
                )
                
        return smoothed

# 사용 예시
smoother = TemporalSmoother(sigma=2.0)
joints_seq = np.array([r['joints'] for r in results])  # (T, 24, 3)
smoothed_joints = smoother.smooth_sequence(joints_seq)
```

---

### Module 5: 운동학 분석 (즉시 제공 가능)

#### 5-1. 관절 각속도 계산
```python
import numpy as np

class KinematicsAnalyzer:
    def __init__(self, fps=30):
        self.fps = fps
        self.dt = 1.0 / fps
        
    def compute_velocity(self, positions):
        """위치 시퀀스 → 속도"""
        velocities = np.gradient(positions, axis=0) / self.dt
        return velocities
    
    def compute_acceleration(self, velocities):
        """속도 시퀀스 → 가속도"""
        accelerations = np.gradient(velocities, axis=0) / self.dt
        return accelerations
    
    def compute_joint_angle(self, j1, j2, j3):
        """3개 관절로 각도 계산 (j2가 꺾이는 점)"""
        v1 = j1 - j2
        v2 = j3 - j2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle_rad)
    
    def analyze_swing(self, joints_sequence):
        """스윙 분석"""
        # 손목 속도 (배트 속도 근사)
        wrist_idx = 21  # SMPL 손목 인덱스
        wrist_positions = joints_sequence[:, wrist_idx, :]
        wrist_velocities = self.compute_velocity(wrist_positions)
        wrist_speeds = np.linalg.norm(wrist_velocities, axis=1)
        
        # 최대 스윙 속도
        max_speed = np.max(wrist_speeds)
        max_speed_frame = np.argmax(wrist_speeds)
        
        # 팔꿈치 각도 (프레임별)
        shoulder_idx = 17
        elbow_idx = 19
        elbow_angles = []
        
        for frame in joints_sequence:
            angle = self.compute_joint_angle(
                frame[shoulder_idx],
                frame[elbow_idx],
                frame[wrist_idx]
            )
            elbow_angles.append(angle)
            
        return {
            'max_swing_speed_ms': max_speed,
            'max_swing_speed_mph': max_speed * 2.237,  # m/s -> mph
            'impact_frame_estimate': max_speed_frame,
            'elbow_angles': elbow_angles
        }

# 사용 예시
analyzer = KinematicsAnalyzer(fps=30)
swing_analysis = analyzer.analyze_swing(smoothed_joints)
print(f"Max Swing Speed: {swing_analysis['max_swing_speed_mph']:.2f} mph")
```

---

## ✅ 제가 바로 제공 가능한 것들

### 1. 스크립트 파일들
- [x] `download_datasets.ps1` - 데이터셋 자동 다운로드
- [x] `setup_hmr.sh` - HMR 환경 설정
- [x] 위의 모든 Python 모듈 코드 (복사 가능)

### 2. 통합 실행 파일 (메인 파이프라인)
```python
# main_pipeline.py - 전체 파이프라인 실행

import argparse
from hmr_inference import HMRInference
from person_detector import PersonDetector
from video_processor import VideoProcessor
from mesh_visualizer import MeshVisualizer
from temporal_smoother import TemporalSmoother
from kinematics_analyzer import KinematicsAnalyzer

def main(args):
    # 1. 모델 초기화
    print("Loading models...")
    detector = PersonDetector()
    hmr_model = HMRInference(args.hmr_model, args.smpl_model)
    
    # 2. 비디오 처리
    print("Processing video...")
    processor = VideoProcessor(hmr_model, detector)
    results = processor.process_video(args.input_video, 'temp_results.npy')
    
    # 3. 시간적 스무딩
    print("Smoothing...")
    smoother = TemporalSmoother(sigma=2.0)
    joints_seq = np.array([r['joints'] for r in results])
    smoothed_joints = smoother.smooth_sequence(joints_seq)
    
    # 4. 운동학 분석
    print("Analyzing kinematics...")
    analyzer = KinematicsAnalyzer(fps=30)
    swing_analysis = analyzer.analyze_swing(smoothed_joints)
    
    # 5. 결과 저장
    print("Saving results...")
    import json
    with open(args.output_json, 'w') as f:
        json.dump(swing_analysis, f, indent=2)
    
    # 6. 3D 시각화
    if args.visualize:
        visualizer = MeshVisualizer(smpl_faces)
        for i, result in enumerate(results[::10]):  # 10프레임마다
            visualizer.save_mesh(
                result['vertices'], 
                f'{args.output_dir}/frame_{i:04d}.obj'
            )
    
    print(f"완료! 결과: {args.output_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', required=True)
    parser.add_argument('--hmr_model', default='models/hmr_model.pt')
    parser.add_argument('--smpl_model', default='models/smpl_neutral.pkl')
    parser.add_argument('--output_json', default='output/analysis.json')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    main(args)
```

---

## ⚠️ 사용자가 직접 수행해야 할 것들

### 1. 수동 다운로드 필요 (라이선스 제약)

#### A. SMPL 모델
- **사이트**: https://smpl.is.tue.mpg.de/
- **절차**:
  1. 회원가입 (이메일 인증)
  2. "Downloads" 페이지
  3. "SMPL for Python" 다운로드
  4. `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` 파일을 `models/` 폴더에 복사

#### B. MPII 데이터셋 (학습 시 필요)
- **사이트**: http://human-pose.mpi-inf.mpg.de/
- **절차**:
  1. 회원가입
  2. `mpii_human_pose_v1.tar.gz` 다운로드
  3. `datasets/mpii/` 폴더에 압축 해제

#### C. Human3.6M 데이터셋 (선택적, fine-tuning 시)
- **사이트**: http://vision.imar.ro/human3.6m/
- **절차**:
  1. 연구자 계정 신청 (승인 1-2일)
  2. Subjects S1~S11 다운로드 (100GB+)
  3. `datasets/h36m/` 폴더에 저장

### 2. 환경 설정 실행
```bash
# Linux/Mac
chmod +x setup_hmr.sh
./setup_hmr.sh

# Windows (PowerShell)
.\download_datasets.ps1
```

### 3. 야구 영상 준비
- 타자 영상 수집 (직접 촬영 or 유튜브)
- 권장 사양:
  - 해상도: 1080p 이상
  - FPS: 30 이상
  - 타자 전신 포함
  - 배경 단순할수록 좋음

### 4. 배트/공 라벨링 (YOLOX Fine-tuning용)
- CVAT 등으로 50~100개 프레임 라벨링
- 클래스: `batter`, `bat`, `ball`
- Export: YOLO format

---

## 📦 최종 폴더 구조

```
baseball_3d_analysis/
├── datasets/
│   ├── coco/
│   ├── mpii/
│   ├── up-3d/
│   └── h36m/
├── models/
│   ├── hmr_model.pt
│   ├── smpl_neutral.pkl
│   └── yolox_x.pth
├── src/
│   ├── hmr_inference.py
│   ├── person_detector.py
│   ├── video_processor.py
│   ├── mesh_visualizer.py
│   ├── temporal_smoother.py
│   ├── kinematics_analyzer.py
│   └── main_pipeline.py
├── output/
│   ├── analysis.json
│   └── meshes/
├── download_datasets.ps1
├── setup_hmr.sh
└── README.md
```

---

## 🚀 실행 순서 요약

### 즉시 실행 가능 (제공된 파일들)
1. `download_datasets.ps1` 실행 → COCO, UP-3D 자동 다운로드
2. 위의 Python 코드들을 `src/` 폴더에 복사

### 사용자 수동 작업
3. SMPL 모델 다운로드 및 설치
4. MPII 회원가입 및 다운로드 (선택)
5. Human3.6M 신청 (선택)
6. 야구 영상 준비
7. (선택) 배트/공 라벨링

### 실행
8. `setup_hmr.sh` 실행 → 환경 구축
9. `python src/main_pipeline.py --input_video baseball.mp4 --visualize`

---

## 💬 다음 단계 선택지

**Q1. 어떤 것부터 시작할까요?**
- A. 우선 HMR 추론 테스트 (단일 이미지)
- B. 전체 파이프라인 코드 먼저 작성
- C. 데이터셋 다운로드부터

**Q2. 사용 환경은?**
- A. 로컬 PC (GPU 있음)
- B. Google Colab
- C. 클라우드 서버

**Q3. 즉시 필요한 코드는?**
- A. 위의 모든 Python 파일을 바로 생성
- B. 특정 모듈만 먼저 (어떤 것?)
- C. 전체 통합 파일 하나로

선택해주시면 해당 부분을 바로 구현해드리겠습니다!
