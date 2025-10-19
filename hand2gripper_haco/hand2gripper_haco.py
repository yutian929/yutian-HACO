import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from .lib.core.config import cfg, update_config
from .lib.models.model import HACO
from .lib.utils.human_models import mano
from .lib.utils.contact_utils import get_contact_thres
from .lib.utils.vis_utils import ContactRenderer, draw_landmarks_on_image, draw_landmarks_on_image_simple
from .lib.utils.preprocessing import augmentation_contact
from .lib.utils.demo_utils import remove_small_contact_components, run_wilor_hand_detector


class HACOContactEstimator:
    """
    HACO model for hand-object contact estimation
    """
    
    def __init__(self, backbone='hamer', checkpoint_path='', experiment_dir='experiments_demo_image'):
        """
        Initialize HACO contact estimator
        
        Args:
            backbone (str): Backbone model type ('hamer', 'vit-l-16', 'vit-b-16', etc.)
            checkpoint_path (str): Path to model checkpoint
            experiment_dir (str): Experiment directory for config
        """
        self.backbone = backbone
        self.checkpoint_path = checkpoint_path
        self.experiment_dir = experiment_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        breakpoint()
        # Load config
        update_config(backbone_type=self.backbone, exp_dir=self.experiment_dir)
        
        # Initialize model
        self.model = HACO().to(self.device)
        self.model.eval()
        
        # Load checkpoint if provided
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded checkpoint from: {self.checkpoint_path}")
        
        # Initialize renderer
        self.contact_renderer = ContactRenderer()
        
        print(f"HACO Contact Estimator initialized with backbone: {self.backbone}")
    
    def preprocess_image(self, image, bbox):
        """
        Preprocess image for model input
        
        Args:
            image (np.ndarray): Input image
            bbox (list): Hand bounding box [x, y, w, h]
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Image preprocessing
        crop_img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(
            image.copy(), bbox, 'test', enforce_flip=False
        )
        
        # Convert to model input format
        if self.backbone in ['handoccnet'] or 'resnet' in cfg.MODEL.backbone_type or 'hrnet' in cfg.MODEL.backbone_type:
            from torchvision import transforms
            img_tensor = transforms.ToTensor()(crop_img.astype(np.float32) / 255.0)
        elif self.backbone in ['hamer'] or 'vit' in cfg.MODEL.backbone_type:
            from torchvision.transforms import Normalize
            normalize = Normalize(mean=cfg.MODEL.img_mean, std=cfg.MODEL.img_std)
            img_tensor = crop_img.transpose(2, 0, 1) / 255.0
            img_tensor = normalize(torch.from_numpy(img_tensor)).float()
        else:
            raise NotImplementedError(f"Unsupported backbone: {self.backbone}")
        
        return img_tensor, crop_img
    
    def predict_contact(self, image, bbox):
        """
        Predict hand-object contact from image and bounding box
        
        Args:
            image (np.ndarray): Input image
            bbox (list): Hand bounding box [x, y, w, h]
            
        Returns:
            dict: Contact prediction results
        """
        # Preprocess image
        img_tensor, crop_img = self.preprocess_image(image, bbox)
        
        # Run model
        with torch.no_grad():
            outputs = self.model({'input': {'image': img_tensor[None].to(self.device)}}, mode="test")
        
        # Process contact output
        eval_thres = get_contact_thres(self.backbone)
        contact_mask = (outputs['contact_out'].sigmoid()[0] > eval_thres).detach().cpu().numpy()
        contact_mask = remove_small_contact_components(contact_mask, faces=mano.watertight_face['right'], min_size=20)
        
        # Render contact visualization
        contact_rendered = self.contact_renderer.render_contact(crop_img[..., ::-1], contact_mask)
        
        return {
            'contact_mask': contact_mask,
            'contact_rendered': contact_rendered,
            'crop_img': crop_img,
            'raw_outputs': outputs
        }
    
    def save_results(self, results, frame_name_base, output_dir='outputs'):
        """
        Save contact estimation results
        
        Args:
            results (dict): Results from predict_contact
            frame_name_base (str): Base name for output files
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/crop_img', exist_ok=True)
        os.makedirs(f'{output_dir}/contact', exist_ok=True)
        
        cv2.imwrite(f'{output_dir}/crop_img/{frame_name_base}.png', results['crop_img'][..., ::-1])
        cv2.imwrite(f'{output_dir}/contact/{frame_name_base}.png', results['contact_rendered'])


class WILORHandDetector:
    """
    WILOR model for hand detection
    """
    
    def __init__(self, detector_type='wilor', detector_path='data/base_data/demo_data/wilor_detector.pt'):
        """
        Initialize WILOR hand detector
        
        Args:
            detector_type (str): Detector type ('wilor' or 'mediapipe')
            detector_path (str): Path to detector model
        """
        self.detector_type = detector_type
        self.detector_path = detector_path
        self.detector = None
        breakpoint()
        if self.detector_type == 'wilor':
            from ultralytics import YOLO
            self.detector = YOLO(self.detector_path)
        elif self.detector_type == 'mediapipe':
            base_options = BaseOptions(model_asset_path=cfg.MODEL.hand_landmarker_path)
            hand_options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
            self.detector = vision.HandLandmarker.create_from_options(hand_options)
        else:
            raise NotImplementedError(f"Unsupported detector: {self.detector_type}")
        
        print(f"WILOR Hand Detector initialized with type: {self.detector_type}")
    
    def detect_hand(self, image):
        """
        Detect hand in image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            tuple: (annotated_image, hand_bbox)
        """
        if self.detector_type == 'wilor':
            right_hand_bbox = run_wilor_hand_detector(image, self.detector)
            annotated_image, right_hand_bbox = draw_landmarks_on_image_simple(image.copy(), right_hand_bbox)
        elif self.detector_type == 'mediapipe':
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image.copy())
            detection_result = self.detector.detect(mp_image)
            annotated_image, right_hand_bbox = draw_landmarks_on_image(image.copy(), detection_result)
        else:
            raise NotImplementedError(f"Unsupported detector: {self.detector_type}")
        
        return annotated_image, right_hand_bbox
    
    def save_detection_results(self, annotated_image, frame_name_base, output_dir='outputs'):
        """
        Save hand detection results
        
        Args:
            annotated_image (np.ndarray): Annotated image with detections
            frame_name_base (str): Base name for output files
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/detection', exist_ok=True)
        
        cv2.imwrite(f'{output_dir}/detection/{frame_name_base}.png', 
                   cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
