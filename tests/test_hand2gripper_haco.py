import pytest
import os
from hand2gripper_haco import HACOContactEstimator, WILORHandDetector

def test_haco_model():
    # Hand Detection Configuration
    detector_type = 'wilor'  # Options: 'wilor' or 'mediapipe'
    # - 'wilor': Uses YOLO-based WILOR detector for hand detection
    # - 'mediapipe': Uses Google MediaPipe for hand landmark detection
    
    detector_path = os.path.join(os.environ['HACO_BASE_DATA_PATH'], 'demo_data', 'wilor_detector.pt')  # Path to WILOR detector model
    # - Required when detector_type='wilor'
    # - Should point to the trained YOLO model file (.pt)
    # - Default: 'data/base_data/demo_data/wilor_detector.pt'
    
    # Contact Estimation Configuration
    backbone = 'hamer'  # Options: 'hamer', 'vit-l-16', 'vit-b-16', 'vit-s-16', 'handoccnet', 'hrnet-w48', 'hrnet-w32', 'resnet-152', 'resnet-101', 'resnet-50', 'resnet-34', 'resnet-18'
    # - 'hamer': Vision Transformer backbone (recommended, best performance)
    # - 'vit-*': Different Vision Transformer variants
    # - 'resnet-*': ResNet backbone variants
    # - 'hrnet-*': High-Resolution Network variants
    # - 'handoccnet': Hand occlusion network
    
    checkpoint_path = os.path.join(os.environ['HACO_BASE_DATA_PATH'], 'release_checkpoint', 'haco_final_hamer_checkpoint.ckpt')  # Path to HACO model checkpoint
    # - Path to the trained HACO model file (.pth or .ckpt)
    # - Leave empty '' if no checkpoint is available (will use random weights)
    # - Example: 'checkpoints/haco_hamer_best.pth'
    
    experiment_dir = 'experiments_demo_image'  # Experiment directory for configuration
    # - Directory where experiment configurations and logs are stored
    # - Used for model configuration and logging
    # - Default: 'experiments_demo_image'
    # breakpoint()
    print("Initializing hand detector...")
    # Initialize WILOR hand detector
    # Parameters:
    # - detector_type: Type of detector ('wilor' or 'mediapipe')
    # - detector_path: Path to detector model file
    hand_detector = WILORHandDetector(
        detector_type=detector_type,
        detector_path=detector_path
    )
    
    print("Initializing contact estimator...")
    # Initialize HACO contact estimator
    # Parameters:
    # - backbone: Model backbone type (see options above)
    # - checkpoint_path: Path to trained model checkpoint
    # - experiment_dir: Directory for experiment configuration
    contact_estimator = HACOContactEstimator(
        backbone=backbone,
        checkpoint_path=checkpoint_path,
        experiment_dir=experiment_dir
    )
    
    print("All components initialized successfully!")