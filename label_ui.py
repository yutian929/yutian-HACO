"""
Label UI for HACO Contact Estimation

Gradio-based Web UI to process images: detect hands and estimate contact.
"""

import os
import cv2
import json
import argparse
import gradio as gr
import numpy as np
import tempfile
from pathlib import Path

from lib.core.config import update_config
from hand_detector import HandDetector
from contact_generator import ContactGenerator


class LabelUIApp:
    """Gradio-based Label UI Application."""
    
    def __init__(self, input_dir: str, output_dir: str = 'outputs'):
        """
        Initialize the Label UI application.
        
        Args:
            input_dir: Path to input folder containing images
            output_dir: Directory to save results
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.current_idx = 0
        
        # Initialize models
        print("Initializing models...")
        update_config(backbone_type='hamer', exp_dir='experiments_demo_image')
        
        self.hand_detector = HandDetector(
            detector_type='wilor',
            detector_path=os.path.join(os.path.dirname(__file__), 
                                      'data', 'base_data', 'demo_data', 
                                      'wilor_detector.pt')
        )
        
        self.contact_generator = ContactGenerator(
            backbone='hamer',
            checkpoint=os.path.join(os.path.dirname(__file__), 
                                   'release_checkpoint', 
                                   'haco_final_hamer_checkpoint.ckpt')
        )
        
        # Get all images
        self.image_files = sorted([
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {input_dir}")
        
        print(f"Found {len(self.image_files)} images")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.detection_dir = os.path.join(output_dir, 'detection')
        self.crop_img_dir = os.path.join(output_dir, 'crop_img')
        self.contact_rendered_dir = os.path.join(output_dir, 'contact_rendered')
        self.gripper_dir = os.path.join(output_dir, 'gripper')
        self.contact_logits_21_dir = os.path.join(output_dir, 'contact_logits_21')
        self.contact_logits_778_dir = os.path.join(output_dir, 'contact_logits_778')
        self.mesh_3d_dir = os.path.join(output_dir, 'mesh_3d')
        self.vertex_coords_dir = os.path.join(output_dir, 'vertex_coords')
        
        os.makedirs(self.detection_dir, exist_ok=True)
        os.makedirs(self.crop_img_dir, exist_ok=True)
        os.makedirs(self.contact_rendered_dir, exist_ok=True)
        os.makedirs(self.gripper_dir, exist_ok=True)
        os.makedirs(self.contact_logits_21_dir, exist_ok=True)
        os.makedirs(self.contact_logits_778_dir, exist_ok=True)
        os.makedirs(self.mesh_3d_dir, exist_ok=True)
        os.makedirs(self.vertex_coords_dir, exist_ok=True)
        
        # Create temp directory for current 3D mesh display
        self.temp_dir = tempfile.mkdtemp(prefix='haco_3d_')
        
        # Cache for current results
        self.current_results = None
        
        # Gripper vertex IDs (0-777)
        self.left_griper_vertex_id = None
        self.right_griper_vertex_id = None
    
    def process_current_image(self):
        """Process the current image and return results."""
        if self.current_idx >= len(self.image_files):
            return None, None, "All images processed!", self._get_progress()
        
        img_file = self.image_files[self.current_idx]
        image_path = os.path.join(self.input_dir, img_file)
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            return None, None, f"❌ Cannot load: {img_file}", self._get_progress()
        
        orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hand
        annotated_image, bbox = self.hand_detector.detect(orig_img)
        
        if bbox is None:
            return orig_img, None, f"❌ No hand detected in: {img_file}", self._get_progress()
        
        # Generate contact estimation
        result = self.contact_generator.single_process(orig_img, bbox, hand_side='right')
        
        # Get vertex coordinates (778, 3)
        vertex_coords = self.contact_generator.get_vertex_coordinates()
        
        # Save 3D mesh to temp file for display (with gripper highlights if selected)
        temp_mesh_path = os.path.join(self.temp_dir, f'current_mesh.glb')
        _, glb_path = self.contact_generator.create_3d_mesh(
            result['contact_mask'], 
            output_path=temp_mesh_path,
            left_vertex_id=self.left_griper_vertex_id,
            right_vertex_id=self.right_griper_vertex_id
        )
        
        # Convert for display (BGR to RGB)
        detection_rgb = annotated_image
        
        num_contact = result['contact_mask'].sum()
        num_vertices = vertex_coords.shape[0]
        info = f"✓ {img_file}\nContact vertices: {num_contact}\nTotal vertices: {num_vertices}"
        
        # Cache results for saving
        self.current_results = {
            'image_name': os.path.splitext(img_file)[0],
            'detection': cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
            'crop': result['crop_img'],
            'contact_rendered': result['contact_rendered'],
            'contact_logits_778': result['outputs']['contact_out'].detach().cpu().numpy(),
            'contact_logits_21': result['outputs']['contact_joint_out'].detach().cpu().numpy(),
            'contact_mask': result['contact_mask'],
            'vertex_coords': vertex_coords
        }
        
        return detection_rgb, glb_path, info, self._get_progress()
    
    def next_image(self):
        """Move to next image."""
        self.current_idx += 1
        return self.process_current_image()
    
    def previous_image(self):
        """Move to previous image."""
        if self.current_idx > 0:
            self.current_idx -= 1
        return self.process_current_image()
    
    def update_gripper_vertex_ids(self, left_vertex_id, right_vertex_id):
        """Update gripper vertex IDs and refresh 3D visualization."""
        # Update vertex IDs
        try:
            if left_vertex_id is not None and left_vertex_id != "":
                left_id = int(left_vertex_id)
                if 0 <= left_id < 778:
                    self.left_griper_vertex_id = left_id
                else:
                    return None, "❌ Left vertex ID must be 0-777"
            else:
                self.left_griper_vertex_id = None
                
            if right_vertex_id is not None and right_vertex_id != "":
                right_id = int(right_vertex_id)
                if 0 <= right_id < 778:
                    self.right_griper_vertex_id = right_id
                else:
                    return None, "❌ Right vertex ID must be 0-777"
            else:
                self.right_griper_vertex_id = None
        except ValueError:
            return None, "❌ Invalid vertex ID format"
        
        # Regenerate 3D mesh with highlights
        if self.current_results is not None:
            temp_mesh_path = os.path.join(self.temp_dir, f'current_mesh.glb')
            _, glb_path = self.contact_generator.create_3d_mesh(
                self.current_results['contact_mask'],
                output_path=temp_mesh_path,
                left_vertex_id=self.left_griper_vertex_id,
                right_vertex_id=self.right_griper_vertex_id
            )
            
            status_msg = f"✓ Gripper vertices updated: Left={self.left_griper_vertex_id} (🔴), Right={self.right_griper_vertex_id} (🔵)"
            return glb_path, status_msg
        
        return None, "No image loaded"
    
    def save_current(self, left_vertex_id, right_vertex_id):
        """Save current results."""
        if self.current_results is None:
            return "❌ No results to save"
        
        # Update gripper vertex IDs (validation done in update method)
        _, update_msg = self.update_gripper_vertex_ids(left_vertex_id, right_vertex_id)
        if "❌" in update_msg:
            return update_msg
        
        try:
            name = self.current_results['image_name']
            
            # Save images
            cv2.imwrite(
                os.path.join(self.detection_dir, f'{name}.png'),
                self.current_results['detection']
            )
            cv2.imwrite(
                os.path.join(self.crop_img_dir, f'{name}.png'),
                self.current_results['crop'][..., ::-1]
            )
            cv2.imwrite(
                os.path.join(self.contact_rendered_dir, f'{name}.png'),
                self.current_results['contact_rendered']
            )
            
            # Save contact estimation logits (778 vertices)
            np.save(
                os.path.join(self.contact_logits_778_dir, f'{name}.npy'),
                self.current_results['contact_logits_778']
            )
            np.save(
                os.path.join(self.contact_logits_21_dir, f'{name}.npy'),
                self.current_results['contact_logits_21']
            )
            
            # Save vertex coordinates (778, 3)
            np.save(
                os.path.join(self.vertex_coords_dir, f'{name}.npy'),
                self.current_results['vertex_coords']
            )
            
            # Save 3D mesh with gripper highlights
            mesh_path = os.path.join(self.mesh_3d_dir, f'{name}.glb')
            self.contact_generator.create_3d_mesh(
                self.current_results['contact_mask'],
                output_path=mesh_path,
                left_vertex_id=self.left_griper_vertex_id,
                right_vertex_id=self.right_griper_vertex_id
            )

            # Save gripper vertex IDs and coordinates to JSON file
            gripper_data = {
                'left_griper_vertex_id': self.left_griper_vertex_id,
                'right_griper_vertex_id': self.right_griper_vertex_id
            }
            
            # Add 3D coordinates of selected vertices
            if self.left_griper_vertex_id is not None:
                gripper_data['left_griper_coords'] = self.current_results['vertex_coords'][self.left_griper_vertex_id].tolist()
            if self.right_griper_vertex_id is not None:
                gripper_data['right_griper_coords'] = self.current_results['vertex_coords'][self.right_griper_vertex_id].tolist()
            
            with open(os.path.join(self.gripper_dir, f'{name}.json'), 'w') as f:
                json.dump(gripper_data, f, indent=4)
            
            return f"✓ Saved: {name}\nLeft vertex: {self.left_griper_vertex_id}, Right vertex: {self.right_griper_vertex_id}"
        except Exception as e:
            return f"❌ Save failed: {str(e)}"
    
    def _get_progress(self):
        """Get progress string."""
        return f"Image {self.current_idx + 1} / {len(self.image_files)}"
    
    def launch(self, share=False):
        """Launch the Gradio interface."""
        with gr.Blocks(title="HACO Label UI") as demo:
            gr.Markdown("# 🖐️ HACO Contact Estimation Label UI")
            gr.Markdown(f"**Input folder:** `{self.input_dir}` | **Output folder:** `{self.output_dir}`")
            
            with gr.Row():
                progress_text = gr.Textbox(
                    label="Progress", 
                    value=self._get_progress(),
                    interactive=False
                )
            
            with gr.Row():
                # Left: Detection
                with gr.Column():
                    detection_img = gr.Image(
                        label="Hand Detection (BBox)",
                        type="numpy"
                    )
                
                # Right: 3D Model
                with gr.Column():
                    mesh_3d = gr.Model3D(
                        label="3D Contact Mesh (Interactive)",
                        clear_color=[1.0, 1.0, 1.0, 1.0],
                        camera_position=(0, 90, 2.5)
                    )
            
            # Info text
            info_text = gr.Textbox(
                label="Info",
                lines=3,
                interactive=False
            )
            
            # Gripper vertex ID selection
            gr.Markdown("### Gripper Vertex Selection (ID: 0-777)")
            gr.Markdown("🔴 **Red** = Left Gripper | 🔵 **Blue** = Right Gripper | 🟢 **Green** = Contact | ⚪ **Gray** = No contact")
            
            with gr.Row():
                left_griper_vertex_input = gr.Number(
                    label="Left Gripper Vertex ID (0-777)",
                    value=None,
                    precision=0,
                    minimum=0,
                    maximum=777,
                    interactive=True
                )
                right_griper_vertex_input = gr.Number(
                    label="Right Gripper Vertex ID (0-777)",
                    value=None,
                    precision=0,
                    minimum=0,
                    maximum=777,
                    interactive=True
                )
            
            # Update button to refresh 3D mesh with highlights
            with gr.Row():
                update_gripper_btn = gr.Button("🔄 Update 3D Visualization", variant="secondary")
            
            # Update status
            update_status = gr.Textbox(
                label="Update Status",
                interactive=False
            )
            
            # Control buttons
            with gr.Row():
                prev_btn = gr.Button("⬅️ Previous", variant="secondary")
                save_btn = gr.Button("💾 Save", variant="primary")
                next_btn = gr.Button("Next ➡️", variant="primary")
            
            # Save status
            save_status = gr.Textbox(
                label="Save Status",
                interactive=False
            )
            
            # Button events
            next_btn.click(
                fn=self.next_image,
                outputs=[detection_img, mesh_3d, info_text, progress_text]
            )
            
            prev_btn.click(
                fn=self.previous_image,
                outputs=[detection_img, mesh_3d, info_text, progress_text]
            )
            
            # Update 3D visualization with selected vertices
            update_gripper_btn.click(
                fn=self.update_gripper_vertex_ids,
                inputs=[left_griper_vertex_input, right_griper_vertex_input],
                outputs=[mesh_3d, update_status]
            )
            
            save_btn.click(
                fn=self.save_current,
                inputs=[left_griper_vertex_input, right_griper_vertex_input],
                outputs=[save_status]
            )
            
            # Load first image on startup
            demo.load(
                fn=self.process_current_image,
                outputs=[detection_img, mesh_3d, info_text, progress_text]
            )
        
        demo.launch(share=share)


def main():
    parser = argparse.ArgumentParser(description='HACO Label UI - Gradio Web Interface')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input folder containing images')
    parser.add_argument('--output', type=str, default='outputs', 
                       help='Output directory (default: outputs)')
    parser.add_argument('--share', action='store_true',
                       help='Create a public share link')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input folder not found: {args.input}")
    
    if not os.path.isdir(args.input):
        raise ValueError(f"Input must be a directory: {args.input}")
    
    # Create and launch app
    app = LabelUIApp(args.input, args.output)
    app.launch(share=args.share)


if __name__ == '__main__':
    main()
