"""
Video Processor Module
프레임별 3D 복원 파이프라인
"""

import cv2
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import os
import json


class VideoProcessor:
    """비디오 프레임별 처리 및 3D 복원"""
    
    def __init__(self, hmr_model, detector, tracker=None):
        """
        Args:
            hmr_model: HMRInference 인스턴스
            detector: PersonDetector 인스턴스
            tracker: MultiObjectTracker 인스턴스 (선택)
        """
        self.hmr = hmr_model
        self.detector = detector
        self.tracker = tracker
        
    def process_video(
        self, 
        video_path: str, 
        output_dir: str,
        save_interval: int = 1,
        max_frames: int = None,
        visualize: bool = False
    ) -> List[Dict]:
        """
        비디오 전체 처리
        
        Args:
            video_path: 입력 비디오 경로
            output_dir: 출력 디렉토리
            save_interval: 저장 간격 (프레임)
            max_frames: 최대 처리 프레임 수 (None이면 전체)
            visualize: 시각화 프레임 저장 여부
            
        Returns:
            프레임별 결과 리스트
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        results = []
        
        # 시각화용 비디오 writer
        vis_writer = None
        if visualize:
            vis_path = os.path.join(output_dir, 'visualization.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vis_writer = cv2.VideoWriter(vis_path, fourcc, fps, (width, height))
        
        # 프레임별 처리
        for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. 객체 검출
            detections = self.detector.detect(frame, target_classes=['person'])
            
            if not detections:
                continue
            
            # 2. 추적 (옵션)
            if self.tracker:
                detections = self.tracker.update(detections)
            
            # 3. 가장 큰 사람 선택 (타자로 가정)
            batter = self.detector.get_largest_person(detections)
            if not batter:
                continue
            
            # 4. HMR 3D 복원
            pred = self.hmr.predict(frame, batter['bbox'])
            
            # 5. 결과 저장
            frame_result = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps,
                'bbox': batter['bbox'],
                'track_id': batter.get('track_id', 0),
                'vertices': pred['vertices'],
                'joints3d': pred['joints3d'],
                'joints2d': pred['joints2d'],
                'shape': pred['shape'],
                'pose': pred['pose'],
                'cam': pred['cam']
            }
            
            results.append(frame_result)
            
            # 6. 시각화 (옵션)
            if visualize and vis_writer:
                vis_frame = self._visualize_frame(frame, batter, pred)
                vis_writer.write(vis_frame)
        
        cap.release()
        if vis_writer:
            vis_writer.release()
        
        # 결과 저장
        self._save_results(results, output_dir)
        
        print(f"Processed {len(results)} frames")
        print(f"Results saved to {output_dir}")
        
        return results
    
    def _visualize_frame(self, frame: np.ndarray, detection: Dict, prediction: Dict) -> np.ndarray:
        """프레임에 검출 결과와 2D 관절 시각화"""
        vis_frame = frame.copy()
        
        # 바운딩 박스
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 2D 관절
        joints2d = prediction['joints2d']
        
        # SMPL 관절 연결 정의
        skeleton = [
            (0, 1), (0, 2), (0, 3),  # 골반 -> 다리
            (1, 4), (2, 5), (3, 6),  # 무릎
            (4, 7), (5, 8), (6, 9),  # 발목
            (0, 12), (12, 13), (13, 14), (14, 15),  # 척추 -> 머리
            (13, 16), (13, 17),  # 어깨
            (16, 18), (17, 19),  # 팔꿈치
            (18, 20), (19, 21),  # 손목
            (20, 22), (21, 23)   # 손
        ]
        
        # 관절 그리기
        for i, (x, y) in enumerate(joints2d):
            # bbox 좌표계로 변환
            x_abs = int(x1 + x * (x2 - x1) / 224)
            y_abs = int(y1 + y * (y2 - y1) / 224)
            cv2.circle(vis_frame, (x_abs, y_abs), 3, (255, 0, 0), -1)
        
        # 스켈레톤 그리기
        for (j1, j2) in skeleton:
            if j1 < len(joints2d) and j2 < len(joints2d):
                x1_j, y1_j = joints2d[j1]
                x2_j, y2_j = joints2d[j2]
                
                x1_abs = int(x1 + x1_j * (x2 - x1) / 224)
                y1_abs = int(y1 + y1_j * (y2 - y1) / 224)
                x2_abs = int(x1 + x2_j * (x2 - x1) / 224)
                y2_abs = int(y1 + y2_j * (y2 - y1) / 224)
                
                cv2.line(vis_frame, (x1_abs, y1_abs), (x2_abs, y2_abs), (0, 255, 255), 2)
        
        return vis_frame
    
    def _save_results(self, results: List[Dict], output_dir: str):
        """결과를 파일로 저장"""
        
        # 1. JSON 메타데이터 저장
        metadata = {
            'num_frames': len(results),
            'frames': []
        }
        
        for r in results:
            metadata['frames'].append({
                'frame_idx': r['frame_idx'],
                'timestamp': r['timestamp'],
                'bbox': r['bbox'],
                'track_id': r['track_id']
            })
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 2. NumPy 배열로 저장
        # Vertices (T, 6890, 3)
        vertices_seq = np.array([r['vertices'] for r in results])
        np.save(os.path.join(output_dir, 'vertices.npy'), vertices_seq)
        
        # Joints3D (T, 24, 3)
        joints3d_seq = np.array([r['joints3d'] for r in results])
        np.save(os.path.join(output_dir, 'joints3d.npy'), joints3d_seq)
        
        # Shape (T, 10)
        shape_seq = np.array([r['shape'] for r in results])
        np.save(os.path.join(output_dir, 'shape.npy'), shape_seq)
        
        # Pose (T, 72)
        pose_seq = np.array([r['pose'] for r in results])
        np.save(os.path.join(output_dir, 'pose.npy'), pose_seq)
        
        print(f"Saved arrays:")
        print(f"  - vertices: {vertices_seq.shape}")
        print(f"  - joints3d: {joints3d_seq.shape}")
        print(f"  - shape: {shape_seq.shape}")
        print(f"  - pose: {pose_seq.shape}")


class FrameExtractor:
    """비디오에서 특정 프레임 추출"""
    
    @staticmethod
    def extract_frames(
        video_path: str, 
        output_dir: str, 
        interval: int = 30,
        max_frames: int = None
    ):
        """
        일정 간격으로 프레임 추출
        
        Args:
            video_path: 비디오 경로
            output_dir: 출력 디렉토리
            interval: 추출 간격 (프레임)
            max_frames: 최대 추출 개수
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        saved_count = 0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % interval == 0:
                output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.jpg')
                cv2.imwrite(output_path, frame)
                saved_count += 1
                
                if max_frames and saved_count >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        print(f"Extracted {saved_count} frames to {output_dir}")


if __name__ == '__main__':
    # 테스트 코드
    print("Testing Video Processor...")
    
    # 더미 비디오 생성
    test_video_path = '/tmp/test_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 30, (640, 480))
    
    for i in range(90):  # 3초 비디오
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    print(f"Created test video: {test_video_path}")
    
    # 프로세서 테스트 (더미 모델)
    from hmr_inference import HMRInference
    from person_detector import PersonDetector, MultiObjectTracker
    
    hmr_model = HMRInference('dummy.pt', 'dummy.pkl')
    detector = PersonDetector()
    tracker = MultiObjectTracker()
    
    processor = VideoProcessor(hmr_model, detector, tracker)
    
    results = processor.process_video(
        test_video_path,
        '/tmp/test_output',
        max_frames=30,
        visualize=True
    )
    
    print(f"\nProcessed {len(results)} frames")
    print("\nTest passed!")
