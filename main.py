from haco import HACOContactEstimator, WILORHandDetector
import os
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    print("=== HACO Hand Detection and Contact Estimation Demo ===")
    
    # ==================== Configuration Parameters ====================
    
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
    
    # Input/Output Configuration
    input_path = os.path.join(os.environ['HACO_BASE_DATA_PATH'], 'example_images')  # Input directory or single image path
    # - Directory containing images to process (will process all .png, .jpg, .jpeg files)
    # - Or path to a single image file
    # - Default: 'asset/example_images'
    
    output_path = 'outputs'  # Output directory for results
    # - Directory where all results will be saved
    # - Will create subdirectories: detection/, crop_img/, contact/
    # - Default: 'outputs'

    
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
    
    # ==================== Process Images ====================
    
    # Check if input path exists
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist!")
        print("Please provide a valid directory or image path.")
        exit(1)
    
    # Determine if input is a directory or single file
    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        
        # Get all image files in directory
        images = sorted([f for f in os.listdir(input_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not images:
            print(f"No image files found in directory: {input_path}")
            exit(1)
        
        print(f"Found {len(images)} images to process")
        
        # Process each image
        results = []
        for image_name in tqdm(images, desc="Processing images"):
            image_path = os.path.join(input_path, image_name)
            
            # Load and convert image
            frame = cv2.imread(image_path)
            orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_name_base = os.path.splitext(image_name)[0]
            
            print(f"\nProcessing: {image_name}")
            
            # Step 1: Hand Detection
            print("  - Detecting hands...")
            annotated_image, right_hand_bbox = hand_detector.detect_hand(orig_img)
            
            if right_hand_bbox is None:
                print(f"  - Skipping {image_name} - no hand detected.")
                continue
            
            print(f"  - Hand detected with bbox: {right_hand_bbox}")
            
            # Step 2: Contact Estimation
            print("  - Estimating hand-object contact...")
            contact_results = contact_estimator.predict_contact(orig_img, right_hand_bbox)
            
            # Step 3: Save Results
            print("  - Saving results...")
            hand_detector.save_detection_results(annotated_image, frame_name_base, output_path)
            contact_estimator.save_results(contact_results, frame_name_base, output_path)
            
            # Store results
            results.append({
                'frame_name': image_name,
                'hand_bbox': right_hand_bbox,
                'contact_results': contact_results
            })
            
            print(f"  - Completed: {image_name}")
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed {len(results)}/{len(images)} images")
        print(f"Results saved to: {output_path}")
        print(f"  - Detection results: {output_path}/detection/")
        print(f"  - Cropped images: {output_path}/crop_img/")
        print(f"  - Contact visualizations: {output_path}/contact/")
        
    else:
        # Process single image
        print(f"Processing single image: {input_path}")
        
        # Load and convert image
        frame = cv2.imread(input_path)
        orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_name = os.path.basename(input_path)
        frame_name_base = os.path.splitext(frame_name)[0]
        
        print(f"Processing: {frame_name}")
        
        # Step 1: Hand Detection
        print("  - Detecting hands...")
        annotated_image, right_hand_bbox = hand_detector.detect_hand(orig_img)
        
        if right_hand_bbox is None:
            print(f"  - No hand detected in {frame_name}")
            exit(1)
        
        print(f"  - Hand detected with bbox: {right_hand_bbox}")
        
        # Step 2: Contact Estimation
        print("  - Estimating hand-object contact...")
        contact_results = contact_estimator.predict_contact(orig_img, right_hand_bbox)
        
        # Step 3: Save Results
        print("  - Saving results...")
        hand_detector.save_detection_results(annotated_image, frame_name_base, output_path)
        contact_estimator.save_results(contact_results, frame_name_base, output_path)
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed: {frame_name}")
        print(f"Results saved to: {output_path}")
        print(f"  - Detection result: {output_path}/detection/{frame_name_base}.png")
        print(f"  - Cropped image: {output_path}/crop_img/{frame_name_base}.png")
        print(f"  - Contact visualization: {output_path}/contact/{frame_name_base}.png")