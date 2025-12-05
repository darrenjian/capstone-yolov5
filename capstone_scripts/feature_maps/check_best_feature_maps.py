#!/usr/bin/env python3
"""
Check Best Feature Maps - YOLOv5 Feature Map and Grad-CAM Visualization

This script loads a YOLOv5 model and visualizes feature maps along with
Grad-CAM overlays to understand what the model is focusing on.

# Basic usage
python check_best_feature_maps.py --image file1000107_17.jpg --weights ./capstone-yolov5/runs/train/.../best.pt

# With custom options
python check_best_feature_maps.py \
    --image /gpfs/scratch/rrr9340/balanced_yolo_dataset/images/val/IM1-01-02-00015.dcm \
    --yolov5-path /gpfs/home/rrr9340/capstone-yolov5 \
    --weights /gpfs/home/rrr9340/capstone-yolov5/runs/train/exp7/weights/best.pt \
    --device cuda \
    --brightness 1.5 \
    --conf-threshold 0.1 \
    --target-layer 21 \
    --output-dir /outputs

# Skip Grad-CAM (faster, just feature maps)
python check_best_feature_maps.py --image myimage.jpg --weights best.pt --skip-gradcam
"""

import argparse
import os
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
import pydicom
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class YoloConfidenceTarget:
    """Dummy scalar target for YOLOv5 confidence scores."""
    def __call__(self, model_output):
        return model_output[0][:, 4].sum()


def get_hook(feature_maps):
    """Create a hook function to capture feature maps."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        feature_maps.append(output.detach().cpu())
    return hook


def load_model(yolov5_path, weights_path, device='cpu', autoshape=False):
    """Load YOLOv5 model from local path."""
    model = torch.hub.load(
        yolov5_path, 'custom',
        path=weights_path,
        source='local',
        force_reload=True,
        device=device,
        autoshape=autoshape
    )
    model.eval()
    return model


def load_image(image_path):
    """Load image from various formats including DICOM."""
    if image_path.lower().endswith('.dcm'):
        dcm = pydicom.dcmread(image_path)
        img_array = dcm.pixel_array.astype(np.float32)
        
        # Normalize to 0-255 range
        img_array = img_array - img_array.min()
        if img_array.max() > 0:
            img_array = (img_array / img_array.max() * 255)
        img_array = img_array.astype(np.uint8)
        
        img = Image.fromarray(img_array)
    else:
        img = Image.open(image_path)
    
    return img


def preprocess_image(image_path, brightness_factor=1.5, size=(640, 640)):
    """Load and preprocess an image for YOLOv5."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # YOLO expects 3-channel
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    
    img = load_image(image_path)
    bright_img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    input_tensor = transform(bright_img).unsqueeze(0)
    
    return img, bright_img, input_tensor


def register_hooks(model, feature_maps, layers, use_conv_only=True):
    """Register forward hooks on model layers to capture feature maps."""
    if use_conv_only:
        # Import Conv from YOLOv5 models
        try:
            from models.common import Conv
            for idx, layer in enumerate(model.model.model):
                if hasattr(layer, 'forward') and isinstance(layer, Conv):
                    layer.register_forward_hook(get_hook(feature_maps))
                    layers.append(idx)
        except ImportError:
            print("Warning: Could not import Conv from models.common. Registering all layers.")
            for idx, layer in enumerate(model.model.model):
                if hasattr(layer, 'forward'):
                    layer.register_forward_hook(get_hook(feature_maps))
                    layers.append(idx)
    else:
        # Register hooks on all layers (for autoshape=True models)
        for idx, layer in enumerate(model.model.model.model):
            if hasattr(layer, 'forward'):
                layer.register_forward_hook(get_hook(feature_maps))
                layers.append(idx)


def visualize_feature_maps(feature_maps, layers, bright_img, output_path='feature_maps_high_res.png'):
    """Visualize averaged feature maps from all layers."""
    n_cols = 5
    n_rows = int(np.ceil((len(feature_maps) + 1) / n_cols))
    
    plt.figure(figsize=(20, n_rows * 4))
    
    # Plot original image first
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(bright_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    valid_layer_idx = 1
    for idx, fmap in enumerate(feature_maps):
        if fmap.dim() == 4:
            avg_fmap = torch.mean(fmap[0], dim=0).numpy()
            plt.subplot(n_rows, n_cols, valid_layer_idx + 1)
            plt.imshow(avg_fmap, cmap='viridis')
            plt.title(f'Layer {layers[idx]}')
            plt.axis('off')
            valid_layer_idx += 1
        else:
            print(f"Skipping layer {layers[idx]} due to incompatible shape {tuple(fmap.shape)}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Feature maps saved to {output_path}")


def visualize_gradcam_overlays(model, input_tensor, feature_maps, layers, bright_img,
                                target_layer_idx=21, output_path='gradcam_featuremap_overlays.png'):
    """Visualize Grad-CAM overlays on the original image."""
    input_tensor.requires_grad_(True)
    
    n_cols = 5
    n_rows = int(np.ceil((len(layers) + 1) / n_cols))
    
    plt.figure(figsize=(20, n_rows * 4))
    
    # Plot original image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(bright_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Prepare RGB image for overlay
    rgb_img = np.array(bright_img.resize((640, 640)).convert("RGB")) / 255.0
    
    valid_layer_idx = 1
    for idx, layer_idx in enumerate(layers):
        target_layer = [model.model.model[target_layer_idx]]
        cam = GradCAM(model=model, target_layers=target_layer)
        
        try:
            targets = [YoloConfidenceTarget()]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        except Exception as e:
            print(f"Skipping layer {layer_idx} due to Grad-CAM error: {e}")
            continue
        
        fmap = feature_maps[idx][0]
        if fmap.dim() != 3:
            print(f"Skipping layer {layer_idx} due to incompatible feature map shape.")
            continue
        
        # Normalize Grad-CAM
        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
        
        # Resize Grad-CAM to match input image
        resized_cam_input = cv2.resize(grayscale_cam, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # Apply jet colormap and blend
        visualization = show_cam_on_image(rgb_img, resized_cam_input, use_rgb=True)
        
        # Plot
        plt.subplot(n_rows, n_cols, valid_layer_idx + 1)
        plt.imshow(visualization)
        plt.title(f'Grad-CAM on Layer {layer_idx}')
        plt.axis('off')
        valid_layer_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Grad-CAM overlays saved to {output_path}")


def visualize_gradcam_on_featuremaps(model, input_tensor, feature_maps, layers, bright_img,
                                      target_layer_idx=21, output_path='gradcam_on_featuremaps.png'):
    """Visualize Grad-CAM blended with feature maps."""
    n_cols = 5
    n_rows = int(np.ceil((len(layers) + 1) / n_cols))
    
    plt.figure(figsize=(20, n_rows * 4))
    
    # Plot original image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(bright_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    valid_layer_idx = 1
    for idx, layer_idx in enumerate(layers):
        target_layer = [model.model.model[target_layer_idx]]
        cam = GradCAM(model=model, target_layers=target_layer)
        
        try:
            targets = [YoloConfidenceTarget()]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        except Exception as e:
            print(f"Skipping layer {layer_idx} due to Grad-CAM error: {e}")
            continue
        
        fmap = feature_maps[idx][0]
        if fmap.dim() != 3:
            print(f"Skipping layer {layer_idx} due to incompatible feature map shape.")
            continue
        
        # Average across channels
        avg_fmap = torch.mean(fmap, dim=0).numpy()
        
        # Normalize both
        avg_fmap = (avg_fmap - avg_fmap.min()) / (avg_fmap.max() - avg_fmap.min() + 1e-8)
        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
        
        # Resize Grad-CAM to feature map size
        H, W = avg_fmap.shape
        resized_cam = cv2.resize(grayscale_cam, (W, H), interpolation=cv2.INTER_LINEAR)
        resized_cam = (resized_cam - resized_cam.min()) / (resized_cam.max() - resized_cam.min() + 1e-8)
        
        # Convert feature map to RGB
        avg_fmap_rgb = np.stack([avg_fmap] * 3, axis=-1)
        
        # Apply colormap to Grad-CAM
        heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Blend: 70% feature map, 30% Grad-CAM heatmap
        blended = 0.7 * avg_fmap_rgb + 0.3 * heatmap
        blended = np.clip(blended, 0, 1)
        
        # Plot
        plt.subplot(n_rows, n_cols, valid_layer_idx + 1)
        plt.imshow(blended)
        plt.title(f'CAM over FeatureMap {layer_idx}')
        plt.axis('off')
        valid_layer_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Grad-CAM on feature maps saved to {output_path}")


def print_predictions(output, conf_threshold=0.1):
    """Print high-confidence predictions."""
    predictions = output[0]
    # Squeeze batch dimension if present
    if predictions.dim() == 3:
        predictions = predictions.squeeze(0)  # [1, 25200, 6] -> [25200, 6]
    high_confidence_preds = predictions[predictions[:, 4] > conf_threshold]
    print(f"Number of high confidence predictions: {high_confidence_preds.shape[0]}")
    print(high_confidence_preds)


def main():
    parser = argparse.ArgumentParser(description='YOLOv5 Feature Map and Grad-CAM Visualization')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--yolov5-path', type=str, default='./capstone-yolov5', help='Path to YOLOv5 repository')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pt file)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu or cuda)')
    parser.add_argument('--brightness', type=float, default=1.5, help='Brightness enhancement factor')
    parser.add_argument('--conf-threshold', type=float, default=0.1, help='Confidence threshold for predictions')
    parser.add_argument('--target-layer', type=int, default=21, help='Target layer index for Grad-CAM')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for saved figures')
    parser.add_argument('--skip-gradcam', action='store_true', help='Skip Grad-CAM visualizations')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract image filename without extension
    image_basename = os.path.splitext(os.path.basename(args.image))[0]

    # Load model (without autoshape for Grad-CAM compatibility)
    print("Loading YOLOv5 model...")
    model = load_model(args.yolov5_path, args.weights, args.device, autoshape=False)
    
    # Load and preprocess image
    print(f"Loading image: {args.image}")
    _, bright_img, input_tensor = preprocess_image(args.image, args.brightness)
    
    # Register hooks
    feature_maps = []
    layers = []
    register_hooks(model, feature_maps, layers, use_conv_only=True)
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output = model(input_tensor)
    
    # Visualize feature maps
    print("Generating feature map visualization...")
    visualize_feature_maps(
        feature_maps, layers, bright_img,
        output_path=f'{args.output_dir}/{image_basename}_feature_maps_high_res.png'
    )
    
    # Print predictions
    print_predictions(output, args.conf_threshold)
    
    if not args.skip_gradcam:
        # Re-run with gradients enabled for Grad-CAM
        input_tensor.requires_grad_(True)
        
        # Grad-CAM overlays on original image
        print("Generating Grad-CAM overlays on original image...")
        visualize_gradcam_overlays(
            model, input_tensor, feature_maps, layers, bright_img,
            target_layer_idx=args.target_layer,
            output_path=f'{args.output_dir}/{image_basename}_gradcam_featuremap_overlays.png'
        )
        
        # Grad-CAM blended with feature maps
        print("Generating Grad-CAM blended with feature maps...")
        visualize_gradcam_on_featuremaps(
            model, input_tensor, feature_maps, layers, bright_img,
            target_layer_idx=args.target_layer,
            output_path=f'{args.output_dir}/{image_basename}_gradcam_on_featuremaps.png'
        )
    
    print("Done!")


if __name__ == '__main__':
    main()