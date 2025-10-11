import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse

from lib.core.config import cfg, update_config
from lib.models.model import HACO
from lib.utils.human_models import mano
from lib.utils.contact_utils import get_contact_thres
from lib.utils.vis_utils import ContactRenderer
from lib.utils.preprocessing import augmentation_contact
from lib.utils.demo_utils import remove_small_contact_components
import trimesh


class ContactGenerator:
    """
    A class that handles the generation of hand-object contact estimation using the HACO model.
    """

    def __init__(self, backbone: str = 'hamer', checkpoint: str = os.path.join(os.path.dirname(__file__), 'release_checkpoint', 'haco_final_hamer_checkpoint.ckpt')):
        """
        Initializes the ContactGenerator with the specified backbone and checkpoint.
        
        Args:
            backbone (str): The model backbone type (default: 'hamer')
            checkpoint (str): Path to the pre-trained checkpoint (default: '')
        """
        # Set device as CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize the model
        self.model = HACO().to(self.device)
        self.model.eval()

        # Load model checkpoint if provided
        if checkpoint:
            checkpoint = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            raise

        # Initialize renderer
        self.eval_thres = get_contact_thres(backbone)
        self.contact_renderer = ContactRenderer()

    def generate_contact(self, crop_rgb: np.ndarray) -> dict:
        """
        Main function to generate hand-object contact using the HACO model.
        
        Args:
            crop_rgb (np.ndarray): The cropped RGB image of the hand.

        Returns:
            dict: The model outputs containing contact estimations.
        """
        # Preprocess the input image
        if cfg.MODEL.backbone_type in ['resnet', 'hrnet']:
            from torchvision import transforms
            img_tensor = transforms.ToTensor()(crop_rgb.astype(np.float32) / 255.0)
        elif cfg.MODEL.backbone_type in ['hamer', 'vit']:
            from torchvision.transforms import Normalize
            normalize = Normalize(mean=cfg.MODEL.img_mean, std=cfg.MODEL.img_std)
            img_tensor = crop_rgb.transpose(2, 0, 1) / 255.0
            img_tensor = normalize(torch.from_numpy(img_tensor)).float()
        else:
            raise NotImplementedError(f"Unsupported backbone: {cfg.MODEL.backbone_type}")

        # Run the model to generate outputs
        with torch.no_grad():
            outputs = self.model({'input': {'image': img_tensor[None].to(self.device)}}, mode="test")

        return outputs
    
    def _crop_img(self, orig_img, right_hand_bbox):
        crop_img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), right_hand_bbox, 'test', enforce_flip=False)
        return crop_img
    
    def create_3d_mesh(self, contact_mask: np.ndarray, output_path: str = None, 
                       left_vertex_id: int = None, right_vertex_id: int = None):
        """
        Create a 3D mesh with contact visualization.
        
        Args:
            contact_mask: Binary contact mask (778,)
            output_path: Optional path to save the mesh as GLB file
            left_vertex_id: Left gripper vertex ID to highlight in red
            right_vertex_id: Right gripper vertex ID to highlight in blue
            
        Returns:
            mesh: trimesh.Trimesh object with contact colors
            glb_path: Path to saved GLB file (if output_path provided)
        """
        # Create a copy of the hand mesh
        mesh = self.contact_renderer.hand_model_mano.copy()
        
        # Define colors
        default_color = np.array([130, 130, 130, 255])   # Gray
        contact_color = np.array([0, 255, 0, 255])       # Green
        left_gripper_color = np.array([255, 0, 0, 255])  # Red for left gripper
        right_gripper_color = np.array([0, 0, 255, 255]) # Blue for right gripper
        
        # Set vertex colors based on contact mask
        vertex_colors = np.tile(default_color, (mesh.vertices.shape[0], 1))
        vertex_colors[contact_mask == 1] = contact_color
        
        # Highlight gripper vertices with different colors
        if left_vertex_id is not None and 0 <= left_vertex_id < mesh.vertices.shape[0]:
            vertex_colors[left_vertex_id] = left_gripper_color
        
        if right_vertex_id is not None and 0 <= right_vertex_id < mesh.vertices.shape[0]:
            vertex_colors[right_vertex_id] = right_gripper_color
        
        mesh.visual.vertex_colors = vertex_colors
        
        glb_path = None
        if output_path:
            # Save as GLB file
            mesh.export(output_path, file_type='glb')
            glb_path = output_path
            
        return mesh, glb_path
    
    def get_vertex_coordinates(self):
        """
        Get the 3D coordinates of all 778 MANO vertices.
        
        Returns:
            vertices: np.ndarray of shape (778, 3) containing xyz coordinates
        """
        return self.contact_renderer.hand_model_mano.vertices.copy()
    
    def single_process(self, orig_img: np.ndarray, right_hand_bbox: np.ndarray, hand_side: str):
        crop_img = self._crop_img(orig_img, right_hand_bbox)
        outputs = self.generate_contact(crop_img)
        contact_mask = (outputs['contact_out'].sigmoid()[0] > self.eval_thres).detach().cpu().numpy()
        contact_mask = remove_small_contact_components(contact_mask, faces=mano.watertight_face[hand_side], min_size=20)
        contact_rendered = self.contact_renderer.render_contact(crop_img[..., ::-1], contact_mask)
        return {
            'crop_img': crop_img,
            'contact_mask': contact_mask,
            'outputs': outputs,
            'contact_rendered': contact_rendered
        }
    


def process_one_demo(input_path: str, bbox_data_path: str, checkpoint_path: str, output_path: str, backbone: str = 'hamer'):
    """
    Process a single demo, generating contact estimation and saving results.

    Args:
        input_path (str): Path to the input images.
        bbox_data_path (str): Path to the .npz file containing pre-detected hand bbox data.
        checkpoint_path (str): Path to the pre-trained HACO model checkpoint.
        output_path (str): Path to save the output results.
        backbone (str): Backbone model type (default: 'hamer').
    """
    # Initialize ContactGenerator
    contact_generator = ContactGenerator(backbone=backbone, checkpoint=checkpoint_path)

    # Load pre-detected bbox data (similar to bbox_processor.py format)
    bbox_data = np.load(bbox_data_path)
    
    # Extract bbox arrays (matching the format saved by demo.py)
    right_hand_detected = bbox_data["right_hand_detected"]  # Boolean array
    right_hand_bboxes = bbox_data["right_bboxes"]  # Shape: (num_frames, 4) - [x1, y1, x2, y2]
    right_hand_bboxes_ctr = bbox_data["right_bboxes_ctr"]  # Shape: (num_frames, 2) - [x, y]

    # Load demo images
    images = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/detection", exist_ok=True)
    os.makedirs(f"{output_path}/crop_img", exist_ok=True)
    os.makedirs(f"{output_path}/contact", exist_ok=True)

    # Loop through images and process
    for i, frame_name in tqdm(enumerate(images), total=len(images)):
        print(f"Processing: {frame_name}")

        # Load and convert image
        frame_path = os.path.join(input_path, frame_name)
        frame = cv2.imread(frame_path)
        orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_name_base = os.path.splitext(frame_name)[0]

        # Get right hand bbox (directly from the .npz file)
        if not right_hand_detected[i]:
            print(f"Skipping {frame_name} - no hand detected.")
            continue
        
        right_hand_bbox = right_hand_bboxes[i]

        print(f"Frame {i}: Right hand bbox: {right_hand_bbox}")

        # Image preprocessing
        crop_img = contact_generator._crop_img(orig_img, right_hand_bbox)

        # Generate contact estimation
        outputs = contact_generator.generate_contact(crop_img)

        # Save results
        cv2.imwrite(f'{output_path}/detection/{frame_name_base}.png', cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{output_path}/crop_img/{frame_name_base}.png', crop_img[..., ::-1])

        eval_thres = get_contact_thres(backbone)
        contact_mask = (outputs['contact_out'].sigmoid()[0] > eval_thres).detach().cpu().numpy()
        contact_mask = remove_small_contact_components(contact_mask, faces=mano.watertight_face['right'], min_size=20)
        contact_rendered = contact_generator.contact_renderer.render_contact(crop_img[..., ::-1], contact_mask)
        cv2.imwrite(f'{output_path}/contact/{frame_name_base}.png', contact_rendered)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Contact Generator Demo')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input images')
    parser.add_argument('--bbox_data_path', type=str, required=True, help='Path to the .npz containing bbox data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output results')
    parser.add_argument('--backbone', type=str, default='hamer', choices=['hamer', 'vit-l-16', 'vit-b-16', 'vit-s-16', 'handoccnet', 'hrnet-w48', 'hrnet-w32', 'resnet-152', 'resnet-101', 'resnet-50', 'resnet-34', 'resnet-18'], help='backbone model')

    args = parser.parse_args()

    # Process the demo
    process_one_demo(args.input_path, args.bbox_data_path, args.checkpoint, args.output_path, args.backbone)
