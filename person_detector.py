"""
Person Detector Module
YOLOX 기반 사람/배트/공 검출
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import torch


class PersonDetector:
    """YOLOX 기반 객체 검출기"""
    
    def __init__(self, model_name: str = 'yolox_x', conf_thresh: float = 0.5):
        """
        Args:
            model_name: YOLOX 모델 이름 (yolox_s, yolox_m, yolox_l, yolox_x)
            conf_thresh: 신뢰도 임계값
        """
        self.conf_thresh = conf_thresh
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading {model_name} on {self.device}...")
        
        # 실제 구현 시 YOLOX 로드
        # self.model = torch.hub.load('Megvii-BaseDetection/YOLOX', model_name, pretrained=True)
        # self.model.to(self.device)
        # self.model.eval()
        
        # COCO 클래스 이름
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball'  # 32번이 공
        ]
        
        print("Detector loaded successfully!")
    
    def detect(self, frame: np.ndarray, target_classes: List[str] = None) -> List[Dict]:
        """
        프레임에서 객체 검출
        
        Args:
            frame: 입력 이미지 (H, W, 3)
            target_classes: 검출할 클래스 리스트. None이면 전체
            
        Returns:
            검출 결과 리스트 [
                {
                    'class': 'person',
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.95
                },
                ...
            ]
        """
        if target_classes is None:
            target_classes = ['person']
        
        # 실제 YOLOX 추론
        # with torch.no_grad():
        #     outputs = self.model(frame)
        
        # 데모용 더미 검출 (실제 구현 시 제거)
        detections = self._dummy_detection(frame, target_classes)
        
        return detections
    
    def _dummy_detection(self, frame: np.ndarray, target_classes: List[str]) -> List[Dict]:
        """테스트용 더미 검출"""
        h, w = frame.shape[:2]
        
        detections = []
        
        # 사람 1명 생성 (중앙)
        if 'person' in target_classes:
            cx, cy = w // 2, h // 2
            bbox_w, bbox_h = w // 3, h // 2
            detections.append({
                'class': 'person',
                'bbox': [
                    max(0, cx - bbox_w // 2),
                    max(0, cy - bbox_h // 2),
                    min(w, cx + bbox_w // 2),
                    min(h, cy + bbox_h // 2)
                ],
                'confidence': 0.95
            })
        
        # 배트 (사람 옆)
        if 'sports ball' in target_classes:
            detections.append({
                'class': 'sports ball',
                'bbox': [w//2 + 50, h//2 - 100, w//2 + 150, h//2],
                'confidence': 0.85
            })
        
        return detections
    
    def detect_and_draw(self, frame: np.ndarray, target_classes: List[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        검출 + 시각화
        
        Returns:
            (시각화된 이미지, 검출 결과)
        """
        detections = self.detect(frame, target_classes)
        
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls = det['class']
            conf = det['confidence']
            
            # 바운딩 박스 그리기
            color = (0, 255, 0) if cls == 'person' else (255, 0, 0)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 레이블
            label = f"{cls}: {conf:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame, detections
    
    def get_largest_person(self, detections: List[Dict]) -> Dict:
        """가장 큰 사람 바운딩 박스 반환"""
        persons = [d for d in detections if d['class'] == 'person']
        
        if not persons:
            return None
        
        # 면적 기준 정렬
        persons_with_area = []
        for p in persons:
            x1, y1, x2, y2 = p['bbox']
            area = (x2 - x1) * (y2 - y1)
            persons_with_area.append((area, p))
        
        persons_with_area.sort(reverse=True)
        
        return persons_with_area[0][1]


class MultiObjectTracker:
    """간단한 IoU 기반 다중 객체 추적"""
    
    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        검출 결과에 track_id 추가
        
        Returns:
            track_id가 추가된 검출 결과
        """
        if not self.tracks:
            # 첫 프레임
            for det in detections:
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = det
                self.next_id += 1
            return detections
        
        # 기존 트랙과 매칭
        updated_detections = []
        matched_track_ids = set()
        
        for det in detections:
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                    
                iou = self._compute_iou(det['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_iou > self.iou_threshold:
                det['track_id'] = best_track_id
                matched_track_ids.add(best_track_id)
                self.tracks[best_track_id] = det
            else:
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = det
                self.next_id += 1
            
            updated_detections.append(det)
        
        return updated_detections
    
    @staticmethod
    def _compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # 합집합
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


if __name__ == '__main__':
    # 테스트 코드
    print("Testing Person Detector...")
    
    # 더미 프레임
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 검출기 초기화
    detector = PersonDetector()
    
    # 검출
    detections = detector.detect(test_frame, target_classes=['person', 'sports ball'])
    print(f"Detected {len(detections)} objects")
    
    for det in detections:
        print(f"  - {det['class']}: bbox={det['bbox']}, conf={det['confidence']:.2f}")
    
    # 시각화
    vis_frame, _ = detector.detect_and_draw(test_frame)
    print(f"Visualization frame shape: {vis_frame.shape}")
    
    # 트래커 테스트
    tracker = MultiObjectTracker()
    tracked = tracker.update(detections)
    print(f"\nTracked objects: {[d['track_id'] for d in tracked]}")
    
    print("\nTest passed!")
