"""
Main Pipeline
야구 타자 3D 복원 및 분석 통합 실행
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path

# 모듈 임포트
from hmr_inference import HMRInference
from person_detector import PersonDetector, MultiObjectTracker
from video_processor import VideoProcessor
from kinematics_analyzer import KinematicsAnalyzer, TemporalSmoother, ViolationDetector

def to_serializable(val):
    # 저장 형식 numpy에서 Native Python으로 변환
    if isinstance(val, (np.int32, np.int64)):
        return int(val)
    if isinstance(val, (np.float32, np.float64)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val

def main(args):
    """메인 파이프라인 실행"""
    
    print("=" * 60)
    print("Baseball 3D Analysis Pipeline")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================
    # Phase 1: 모델 로딩
    # ========================================
    print("\n[Phase 1] Loading models...")
    
    # 1-1. HMR 모델
    print("  - Loading HMR model...")
    hmr_model = HMRInference(args.hmr_model, args.smpl_model)
    
    # 1-2. 객체 검출기
    print("  - Loading detector...")
    detector = PersonDetector(conf_thresh=args.conf_thresh)
    
    # 1-3. 추적기 (옵션)
    tracker = None
    if args.use_tracking:
        print("  - Initializing tracker...")
        tracker = MultiObjectTracker()
    
    print("✓ Models loaded successfully!")
    
    # ========================================
    # Phase 2: 비디오 처리
    # ========================================
    print("\n[Phase 2] Processing video...")
    
    processor = VideoProcessor(hmr_model, detector, tracker)
    
    raw_output_dir = os.path.join(args.output_dir, 'raw')
    results = processor.process_video(
        args.input_video,
        raw_output_dir,
        max_frames=args.max_frames,
        visualize=args.visualize
    )
    
    if len(results) == 0:
        print("✗ No frames processed. Check video and detection settings.")
        return
    
    print(f"✓ Processed {len(results)} frames")
    
    # ========================================
    # Phase 3: 시간적 스무딩
    # ========================================
    print("\n[Phase 3] Temporal smoothing...")
    
    smoother = TemporalSmoother(sigma=args.smooth_sigma)
    
    # Joints 시퀀스 로드
    joints_sequence = np.array([r['joints3d'] for r in results])
    
    # 스무딩
    smoothed_joints = smoother.smooth_sequence(joints_sequence)
    
    # 저장
    np.save(os.path.join(args.output_dir, 'joints3d_smoothed.npy'), smoothed_joints)
    
    print(f"✓ Smoothed joints shape: {smoothed_joints.shape}")
    
    # ========================================
    # Phase 4: 운동학 분석
    # ========================================
    print("\n[Phase 4] Kinematics analysis...")
    
    analyzer = KinematicsAnalyzer(fps=args.fps)
    
    # 4-1. 스윙 분석
    swing_analysis = analyzer.analyze_swing(smoothed_joints)
    
    print(f"  - Max Swing Speed: {swing_analysis['max_swing_speed_mph']:.2f} mph")
    print(f"  - Impact Frame: {swing_analysis['impact_frame_estimate']}")
    print(f"  - Swing Duration: {swing_analysis['swing_duration_s']:.3f}s")
    
    # 4-2. 동작 단계 검출
    phases = analyzer.detect_phases(smoothed_joints)
    
    print(f"  - Detected phases:")
    for phase_name, phase_range in phases.items():
        if isinstance(phase_range, tuple):
            print(f"    • {phase_name}: frames {phase_range[0]}-{phase_range[1]}")
        else:
            print(f"    • {phase_name}: frame {phase_range}")
    
    # 4-3. 배트 궤적 추정
    bat_trajectory = analyzer.compute_bat_trajectory(smoothed_joints, bat_length=0.85)
    np.save(os.path.join(args.output_dir, 'bat_trajectory.npy'), bat_trajectory)
    
    print("✓ Kinematics analysis complete")
    
    # ========================================
    # Phase 5: 반칙 판정
    # ========================================
    print("\n[Phase 5] Violation detection...")
    
    violation_detector = ViolationDetector()
    violations = violation_detector.detect_violations(smoothed_joints, phases)
    
    if violations:
        print(f"  ⚠ Violations detected: {', '.join(violations)}")
    else:
        print(f"  ✓ No violations detected")
    
    # ========================================
    # Phase 6: 결과 저장
    # ========================================
    print("\n[Phase 6] Saving results...")
    
    # 통합 리포트 생성
    report = {
        'video_info': {
            'input_path': args.input_video,
            'total_frames_processed': len(results),
            'fps': args.fps
        },
        'swing_analysis': swing_analysis,
        'phases': {k: (list(v) if isinstance(v, tuple) else int(v)) 
                   for k, v in phases.items()},
        'violations': violations,
        'output_files': {
            'joints3d_raw': os.path.join(raw_output_dir, 'joints3d.npy'),
            'joints3d_smoothed': os.path.join(args.output_dir, 'joints3d_smoothed.npy'),
            'vertices': os.path.join(raw_output_dir, 'vertices.npy'),
            'bat_trajectory': os.path.join(args.output_dir, 'bat_trajectory.npy')
        }
    }
    
    # JSON 저장
    report_path = os.path.join(args.output_dir, 'analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=to_serializable)
    
    print(f"✓ Report saved to: {report_path}")
    
    # ========================================
    # Phase 7: 시각화 (옵션)
    # ========================================
    if args.visualize:
        print("\n[Phase 7] Visualization...")
        print(f"  - Visualization video: {os.path.join(raw_output_dir, 'visualization.mp4')}")
    
    # ========================================
    # 완료
    # ========================================
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"\nResults directory: {args.output_dir}")
    print(f"Analysis report: {report_path}")
    
    # 요약 출력
    print("\n📊 Summary:")
    print(f"  • Frames processed: {len(results)}")
    print(f"  • Max swing speed: {swing_analysis['max_swing_speed_mph']:.2f} mph")
    print(f"  • Impact frame: {swing_analysis['impact_frame_estimate']}")
    print(f"  • Violations: {len(violations)}")


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Baseball 3D Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 입출력
    parser.add_argument('--input_video', type=str, required=True,
                       help='Input video path')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory')
    
    # 모델 경로
    parser.add_argument('--hmr_model', type=str, default='models/hmr_model.pt',
                       help='HMR model checkpoint path')
    parser.add_argument('--smpl_model', type=str, default='models/smpl_neutral.pkl',
                       help='SMPL model path')
    
    # 처리 옵션
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Video FPS')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to process (None for all)')
    parser.add_argument('--conf_thresh', type=float, default=0.5,
                       help='Detection confidence threshold')
    
    # 기능 옵션
    parser.add_argument('--use_tracking', action='store_true',
                       help='Enable multi-object tracking')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization video')
    parser.add_argument('--smooth_sigma', type=float, default=2.0,
                       help='Temporal smoothing sigma')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
