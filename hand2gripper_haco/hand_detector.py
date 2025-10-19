"""
Hand Detector Module

This module provides hand detection capabilities using multiple detector backends
(WiLoR and MediaPipe) for hand localization in images.
"""

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from lib.core.config import cfg
from lib.utils.vis_utils import draw_landmarks_on_image, draw_landmarks_on_image_simple
from lib.utils.demo_utils import run_wilor_hand_detector


class HandDetector:
    """
    A class that handles hand detection using different detector backends.
    
    Supports:
    - WiLoR detector: YOLO-based hand detector
    - MediaPipe detector: Google's MediaPipe hand landmark detector
    """
    
    def __init__(self, detector_type: str = 'wilor', detector_path: str = None):
        """
        Initialize the HandDetector with the specified detector type.
        
        Args:
            detector_type (str): Type of detector - 'wilor' or 'mediapipe'
            detector_path (str): Path to detector model (for wilor only)
        """
        self.detector_type = detector_type
        
        if detector_type == 'wilor':
            from ultralytics import YOLO
            if detector_path is None:
                detector_path = 'data/base_data/demo_data/wilor_detector.pt'
            
            if not os.path.exists(detector_path):
                raise FileNotFoundError(f"WiLoR detector not found: {detector_path}")
            
            self.detector = YOLO(detector_path)
            
        elif detector_type == 'mediapipe':
            if not hasattr(cfg.MODEL, 'hand_landmarker_path'):
                raise ValueError("MediaPipe hand_landmarker_path not configured in cfg.MODEL")
            
            if not os.path.exists(cfg.MODEL.hand_landmarker_path):
                raise FileNotFoundError(f"MediaPipe model not found: {cfg.MODEL.hand_landmarker_path}")
            
            base_options = BaseOptions(model_asset_path=cfg.MODEL.hand_landmarker_path)
            hand_options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
            self.detector = vision.HandLandmarker.create_from_options(hand_options)
            
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
    
    def detect(self, orig_img: np.ndarray) -> tuple:
        """
        Detect hand in the image and return annotated image and bounding box.
        
        Args:
            orig_img (np.ndarray): Input RGB image
            
        Returns:
            tuple: (annotated_image, right_hand_bbox)
                - annotated_image: Image with detection visualization
                - right_hand_bbox: Bounding box [x1, y1, x2, y2] or None if no hand detected
        """
        if self.detector_type == 'wilor':
            right_hand_bbox = run_wilor_hand_detector(orig_img, self.detector)
            annotated_image, right_hand_bbox = draw_landmarks_on_image_simple(
                orig_img.copy(), right_hand_bbox
            )
            
        elif self.detector_type == 'mediapipe':
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=orig_img.copy())
            detection_result = self.detector.detect(mp_image)
            annotated_image, right_hand_bbox = draw_landmarks_on_image(
                orig_img.copy(), detection_result
            )
            
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")
        
        return annotated_image, right_hand_bbox

