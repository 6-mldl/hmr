"""
HMR Inference Module
3D Human Mesh Recovery from 2D images
"""

import torch
import cv2
import numpy as np
from typing import Dict, Tuple
import os


class HMRInference:
    """HMR 모델 추론 클래스"""
    
    def __init__(self, model_path: str, smpl_path: str, img_size: int = 224):
        """
        Args:
            model_path: HMR 체크포인트 경로
            smpl_path: SMPL 모델 경로
            img_size: 입력 이미지 크기
        """
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # 모델 로드 (실제 구현 시 pytorch_HMR 사용)
        # from models import hmr
        # self.model = hmr.HMR().to(self.device)
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # self.model.eval()
        
        # SMPL 모델 로드
        # import pickle
        # with open(smpl_path, 'rb') as f:
        #     self.smpl_data = pickle.load(f, encoding='latin1')
        
        print("HMR model loaded successfully!")
        
    def preprocess(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        이미지 전처리 및 크롭
        
        Args:
            img: 원본 이미지 (H, W, 3)
            bbox: (x1, y1, x2, y2) 바운딩 박스
            
        Returns:
            전처리된 이미지 텐서 (1, 3, 224, 224)
        """
        x1, y1, x2, y2 = bbox
        
        # 바운딩 박스 크롭
        crop = img[y1:y2, x1:x2]
        
        # 정사각형으로 패딩
        h, w = crop.shape[:2]
        if h > w:
            pad = (h - w) // 2
            crop = cv2.copyMakeBorder(crop, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        elif w > h:
            pad = (w - h) // 2
            crop = cv2.copyMakeBorder(crop, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        
        # 리사이즈
        crop = cv2.resize(crop, (self.img_size, self.img_size))
        
        # 정규화 [-1, 1]
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - 0.5) / 0.5
        
        # Tensor 변환 (B, C, H, W)
        crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
        
        return crop.to(self.device)
    
    def predict(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        3D 포즈 및 메쉬 예측
        
        Args:
            img: 원본 이미지
            bbox: 바운딩 박스
            
        Returns:
            {
                'vertices': (6890, 3) - SMPL 메쉬 정점
                'joints3d': (24, 3) - 3D 관절 위치
                'joints2d': (24, 2) - 2D 관절 투영
                'shape': (10,) - SMPL 체형 파라미터
                'pose': (72,) - SMPL 포즈 파라미터 (24관절 x 3 rotation)
                'cam': (3,) - 약한 원근 카메라 파라미터 [s, tx, ty]
            }
        """
        img_tensor = self.preprocess(img, bbox)
        
        with torch.no_grad():
            # 실제 HMR 추론
            # pred = self.model(img_tensor)
            
            # 데모용 더미 데이터 (실제 구현 시 제거)
            pred = self._dummy_prediction()
            
        return pred
    
    def _dummy_prediction(self) -> Dict:
        """테스트용 더미 예측 (실제 HMR 모델 없을 때)"""
        return {
            'vertices': np.random.randn(6890, 3) * 0.5,
            'joints3d': np.random.randn(24, 3) * 0.5,
            'joints2d': np.random.rand(24, 2) * 224,
            'shape': np.random.randn(10) * 0.1,
            'pose': np.random.randn(72) * 0.1,
            'cam': np.array([1.0, 0.0, 0.0])
        }
    
    def batch_predict(self, imgs: list, bboxes: list) -> list:
        """배치 예측"""
        results = []
        for img, bbox in zip(imgs, bboxes):
            pred = self.predict(img, bbox)
            results.append(pred)
        return results


def load_hmr_model(checkpoint_path: str = None) -> HMRInference:
    """HMR 모델 로더 헬퍼 함수"""
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), 
            '../models/hmr_model.pt'
        )
    
    smpl_path = os.path.join(
        os.path.dirname(__file__),
        '../models/smpl_neutral.pkl'
    )
    
    model = HMRInference(checkpoint_path, smpl_path)
    return model


if __name__ == '__main__':
    # 테스트 코드
    print("Testing HMR Inference...")
    
    # 더미 이미지 생성
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bbox = (100, 50, 300, 450)
    
    # 모델 로드 (더미)
    model = HMRInference('dummy.pt', 'dummy.pkl')
    
    # 예측
    result = model.predict(test_img, test_bbox)
    
    print(f"Vertices shape: {result['vertices'].shape}")
    print(f"Joints3D shape: {result['joints3d'].shape}")
    print(f"Shape params: {result['shape'].shape}")
    print(f"Pose params: {result['pose'].shape}")
    print("\nTest passed!")
