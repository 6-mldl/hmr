"""
Kinematics Analyzer Module
3D 관절 데이터 기반 운동학 분석
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple
import json


class KinematicsAnalyzer:
    """운동학 분석 클래스"""
    
    # SMPL 관절 인덱스
    JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ]
    
    # 주요 관절 인덱스
    PELVIS = 0
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    
    def __init__(self, fps: float = 30.0):
        """
        Args:
            fps: 비디오 FPS
        """
        self.fps = fps
        self.dt = 1.0 / fps
        
    def compute_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        위치 시퀀스 → 속도
        
        Args:
            positions: (T, ...) 위치 배열
            
        Returns:
            (T, ...) 속도 배열 (m/s)
        """
        velocities = np.gradient(positions, axis=0) / self.dt
        return velocities
    
    def compute_acceleration(self, velocities: np.ndarray) -> np.ndarray:
        """
        속도 시퀀스 → 가속도
        
        Args:
            velocities: (T, ...) 속도 배열
            
        Returns:
            (T, ...) 가속도 배열 (m/s²)
        """
        accelerations = np.gradient(velocities, axis=0) / self.dt
        return accelerations
    
    def compute_joint_angle(
        self, 
        j1: np.ndarray, 
        j2: np.ndarray, 
        j3: np.ndarray
    ) -> float:
        """
        3개 관절로 각도 계산
        
        Args:
            j1, j2, j3: 관절 3D 좌표 (j2가 꺾이는 점)
            
        Returns:
            각도 (도)
        """
        v1 = j1 - j2
        v2 = j3 - j2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle_rad)
    
    def analyze_swing(self, joints_sequence: np.ndarray) -> Dict:
        """
        스윙 분석
        
        Args:
            joints_sequence: (T, 24, 3) 관절 시퀀스
            
        Returns:
            분석 결과 딕셔너리
        """
        T = len(joints_sequence)
        
        # 1. 손목 속도 분석 (배트 속도 근사)
        wrist_positions = joints_sequence[:, self.RIGHT_WRIST, :]  # 오른손 타자 가정
        wrist_velocities = self.compute_velocity(wrist_positions)
        wrist_speeds = np.linalg.norm(wrist_velocities, axis=1)
        
        # 최대 스윙 속도
        max_speed_idx = np.argmax(wrist_speeds)
        max_speed_ms = wrist_speeds[max_speed_idx]
        max_speed_mph = max_speed_ms * 2.237  # m/s -> mph
        
        # 2. 팔꿈치 각도 (프레임별)
        elbow_angles = []
        for frame in joints_sequence:
            angle = self.compute_joint_angle(
                frame[self.RIGHT_SHOULDER],
                frame[self.RIGHT_ELBOW],
                frame[self.RIGHT_WRIST]
            )
            elbow_angles.append(angle)
        
        # 3. 어깨 회전 (골반 대비)
        shoulder_rotations = []
        for frame in joints_sequence:
            # 어깨 벡터
            shoulder_vec = frame[self.RIGHT_SHOULDER] - frame[self.LEFT_SHOULDER]
            # XY 평면 투영 후 각도
            angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
            shoulder_rotations.append(np.degrees(angle))
        
        # 4. 골반-어깨 분리 (separation)
        pelvis_positions = joints_sequence[:, self.PELVIS, :]
        pelvis_velocities = self.compute_velocity(pelvis_positions)
        
        # 5. 스윙 단계 구분
        speed_threshold = max_speed_ms * 0.3
        swing_start_idx = np.where(wrist_speeds > speed_threshold)[0][0] if np.any(wrist_speeds > speed_threshold) else 0
        swing_duration = (max_speed_idx - swing_start_idx) / self.fps
        
        return {
            'max_swing_speed_ms': float(max_speed_ms),
            'max_swing_speed_mph': float(max_speed_mph),
            'impact_frame_estimate': int(max_speed_idx),
            'impact_time_s': float(max_speed_idx / self.fps),
            'swing_start_frame': int(swing_start_idx),
            'swing_duration_s': float(swing_duration),
            'elbow_angles': elbow_angles,
            'shoulder_rotations': shoulder_rotations,
            'average_elbow_angle': float(np.mean(elbow_angles)),
            'max_shoulder_rotation': float(np.max(np.abs(shoulder_rotations)))
        }
    
    def compute_bat_trajectory(
        self, 
        joints_sequence: np.ndarray,
        bat_length: float = 0.85  # 배트 길이 (미터)
    ) -> np.ndarray:
        """
        배트 끝 궤적 추정
        
        Args:
            joints_sequence: (T, 24, 3)
            bat_length: 배트 길이 (m)
            
        Returns:
            (T, 3) 배트 끝 위치
        """
        wrist_positions = joints_sequence[:, self.RIGHT_WRIST, :]
        elbow_positions = joints_sequence[:, self.RIGHT_ELBOW, :]
        
        # 팔 방향 벡터
        arm_vectors = wrist_positions - elbow_positions
        arm_vectors = arm_vectors / (np.linalg.norm(arm_vectors, axis=1, keepdims=True) + 1e-8)
        
        # 배트 끝 = 손목 + 배트길이 * 팔방향
        bat_tip_positions = wrist_positions + bat_length * arm_vectors
        
        return bat_tip_positions
    
    def detect_phases(self, joints_sequence: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        동작 단계 검출
        
        Returns:
            {
                'stance': (start_frame, end_frame),
                'load': (start_frame, end_frame),
                'swing': (start_frame, end_frame),
                'contact': frame,
                'follow_through': (start_frame, end_frame)
            }
        """
        T = len(joints_sequence)
        
        # 손목 속도 기반 단계 구분
        wrist_positions = joints_sequence[:, self.RIGHT_WRIST, :]
        wrist_velocities = self.compute_velocity(wrist_positions)
        wrist_speeds = np.linalg.norm(wrist_velocities, axis=1)
        
        max_speed_idx = np.argmax(wrist_speeds)
        speed_threshold = wrist_speeds[max_speed_idx] * 0.2
        
        # 스윙 시작
        swing_candidates = np.where(wrist_speeds > speed_threshold)[0]
        swing_start = swing_candidates[0] if len(swing_candidates) > 0 else 0
        
        # 로드 단계 (스윙 전)
        load_start = max(0, swing_start - int(0.3 * self.fps))  # 스윙 0.3초 전
        
        # 컨택트 (최대 속도)
        contact_frame = max_speed_idx
        
        # 팔로우 스루 (컨택트 후)
        follow_through_end = min(T - 1, contact_frame + int(0.5 * self.fps))
        
        return {
            'stance': (0, load_start),
            'load': (load_start, swing_start),
            'swing': (swing_start, contact_frame),
            'contact': contact_frame,
            'follow_through': (contact_frame, follow_through_end)
        }


class TemporalSmoother:
    """시간적 스무딩"""
    
    def __init__(self, sigma: float = 2.0):
        """
        Args:
            sigma: Gaussian 필터 시그마
        """
        self.sigma = sigma
    
    def smooth_sequence(self, data: np.ndarray) -> np.ndarray:
        """
        시퀀스 스무딩
        
        Args:
            data: (T, ...) 시퀀스
            
        Returns:
            스무딩된 시퀀스
        """
        original_shape = data.shape
        
        # (T, ...) → (T, N)
        data_flat = data.reshape(original_shape[0], -1)
        
        smoothed = np.zeros_like(data_flat)
        for i in range(data_flat.shape[1]):
            smoothed[:, i] = gaussian_filter1d(data_flat[:, i], sigma=self.sigma)
        
        # 원래 shape으로 복원
        smoothed = smoothed.reshape(original_shape)
        
        return smoothed


class ViolationDetector:
    """반칙 판정"""
    
    def __init__(self):
        self.batter_box = {
            'x_min': -0.6,  # 미터 단위 (예시)
            'x_max': 0.6,
            'y_min': -1.0,
            'y_max': 1.0
        }
    
    def check_out_of_box(self, joints_sequence: np.ndarray, contact_frame: int) -> bool:
        """배터 박스 이탈 체크"""
        # 컨택트 시점의 골반 위치
        pelvis_at_contact = joints_sequence[contact_frame, 0, :2]  # XY만
        
        out_of_box = (
            pelvis_at_contact[0] < self.batter_box['x_min'] or
            pelvis_at_contact[0] > self.batter_box['x_max'] or
            pelvis_at_contact[1] < self.batter_box['y_min'] or
            pelvis_at_contact[1] > self.batter_box['y_max']
        )
        
        return out_of_box
    
    def detect_violations(
        self, 
        joints_sequence: np.ndarray,
        phases: Dict
    ) -> List[str]:
        """전체 반칙 체크"""
        violations = []
        
        contact_frame = phases['contact']
        
        # 1. 박스 이탈
        if self.check_out_of_box(joints_sequence, contact_frame):
            violations.append("OUT_OF_BOX")
        
        # 2. 기타 반칙들 추가 가능
        # - 배트 던지기
        # - 더블 히트
        # 등등...
        
        return violations


if __name__ == '__main__':
    # 테스트 코드
    print("Testing Kinematics Analyzer...")
    
    # 더미 데이터 생성
    T = 90  # 3초, 30fps
    joints_sequence = np.random.randn(T, 24, 3) * 0.5
    
    # 분석기 초기화
    analyzer = KinematicsAnalyzer(fps=30)
    
    # 스윙 분석
    swing_analysis = analyzer.analyze_swing(joints_sequence)
    
    print(f"Max Swing Speed: {swing_analysis['max_swing_speed_mph']:.2f} mph")
    print(f"Impact Frame: {swing_analysis['impact_frame_estimate']}")
    print(f"Swing Duration: {swing_analysis['swing_duration_s']:.3f}s")
    print(f"Average Elbow Angle: {swing_analysis['average_elbow_angle']:.1f}°")
    
    # 단계 검출
    phases = analyzer.detect_phases(joints_sequence)
    print(f"\nPhases:")
    for phase_name, phase_range in phases.items():
        print(f"  {phase_name}: {phase_range}")
    
    # 스무딩 테스트
    smoother = TemporalSmoother(sigma=2.0)
    smoothed = smoother.smooth_sequence(joints_sequence)
    print(f"\nSmoothed shape: {smoothed.shape}")
    
    # 반칙 검출
    detector = ViolationDetector()
    violations = detector.detect_violations(joints_sequence, phases)
    print(f"\nViolations: {violations if violations else 'None'}")
    
    print("\nTest passed!")
