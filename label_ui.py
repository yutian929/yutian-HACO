"""
Label UI for HACO Contact Estimation

Process images in a folder: detect hands and estimate contact.
"""

import os
import cv2
import argparse
from tqdm import tqdm

from lib.core.config import update_config
from hand_detector import HandDetector
from contact_generator import ContactGenerator


def process_images(input_dir: str, output_dir: str = 'outputs'):
    """
    Process all images in a folder: detect hand and estimate contact.
    
    Args:
        input_dir: Path to input folder containing images
        output_dir: Directory to save results
    """
    # Initialize (use defaults)
    update_config(backbone_type='hamer', exp_dir='experiments_demo_image')
    
    print("Initializing HandDetector...")
    hand_detector = HandDetector(detector_type='wilor', detector_path=os.path.join(os.path.dirname(__file__), 'data', 'base_data', 'demo_data', 'wilor_detector.pt'))
    
    print("Initializing ContactGenerator...")
    contact_generator = ContactGenerator(backbone='hamer', checkpoint=os.path.join(os.path.dirname(__file__), 'release_checkpoint', 'haco_final_hamer_checkpoint.ckpt'))
    
    # Get all images
    image_files = sorted([f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'detection'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'crop_img'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'contact'), exist_ok=True)
    
    # Process each image
    print("\nProcessing images...")
    for img_file in tqdm(image_files):
        image_path = os.path.join(input_dir, img_file)
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  ✗ Cannot load: {img_file}")
            continue
        
        orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hand
        annotated_image, bbox = hand_detector.detect(orig_img)
        
        if bbox is None:
            print(f"  ✗ No hand detected: {img_file}")
            continue
        
        # Generate contact estimation
        result = contact_generator.single_process(orig_img, bbox, hand_side='right')
        
        # Save results
        image_name = os.path.splitext(img_file)[0]
        
        cv2.imwrite(os.path.join(output_dir, 'detection', f'{image_name}.png'),
                   cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, 'crop_img', f'{image_name}.png'),
                   result['crop_img'][..., ::-1])
        cv2.imwrite(os.path.join(output_dir, 'contact', f'{image_name}.png'),
                   result['contact_rendered'])
    
    print(f"\n✓ Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='HACO Label UI - Contact Estimation')
    parser.add_argument('--input', type=str, required=True, help='Path to input folder containing images')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory (default: outputs)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input folder not found: {args.input}")
    
    if not os.path.isdir(args.input):
        raise ValueError(f"Input must be a directory: {args.input}")
    
    process_images(args.input, args.output)
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
