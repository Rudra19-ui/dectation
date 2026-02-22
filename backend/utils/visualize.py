#!/usr/bin/env python3
"""
Visualization Utilities for Breast Cancer Classification

This module provides functions for visualizing model predictions,
including Grad-CAM heatmaps with ROI overlay.
"""

import os
import uuid
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def apply_colormap(heatmap, cmap_name='jet'):
    """
    Apply a colormap to a heatmap.
    
    Args:
        heatmap: Numpy array with values between 0 and 1
        cmap_name: Name of the colormap to use
        
    Returns:
        Colorized heatmap as RGB image
    """
    # Get colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Apply colormap
    colored_heatmap = cmap(heatmap)
    
    # Convert to RGB uint8
    colored_heatmap = np.uint8(colored_heatmap[:, :, :3] * 255)
    
    return colored_heatmap


def generate_roi_mask(image, threshold=0.05):
    """
    Generate a Region of Interest (ROI) mask for breast tissue.
    This is a simplified version - in a real application, more sophisticated
    segmentation methods would be used.
    
    Args:
        image: Input image as numpy array
        threshold: Threshold for binary segmentation
        
    Returns:
        Binary mask of the ROI
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def apply_heatmap_with_roi(image, heatmap, roi_mask=None, alpha=0.5):
    """
    Apply heatmap overlay on the original image, constrained by ROI mask.
    
    Args:
        image: Original image as numpy array (RGB)
        heatmap: Heatmap as numpy array (values between 0 and 1)
        roi_mask: Binary mask for the region of interest (optional)
        alpha: Transparency of the heatmap overlay
        
    Returns:
        Image with heatmap overlay
    """
    # Ensure image is RGB
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize heatmap to match image dimensions if needed
    if image.shape[:2] != heatmap.shape:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Generate ROI mask if not provided
    if roi_mask is None:
        roi_mask = generate_roi_mask(image)
    
    # Resize mask to match image dimensions if needed
    if image.shape[:2] != roi_mask.shape:
        roi_mask = cv2.resize(roi_mask, (image.shape[1], image.shape[0]))
    
    # Apply ROI mask to heatmap
    masked_heatmap = heatmap * roi_mask
    
    # Apply colormap to heatmap
    colored_heatmap = apply_colormap(masked_heatmap)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)
    
    return overlay


def save_visualization(image, output_path=None, output_dir="reports"):
    """
    Save visualization to file.
    
    Args:
        image: Image to save (numpy array or PIL Image)
        output_path: Path to save the image (optional)
        output_dir: Directory to save the image if output_path is not provided
        
    Returns:
        Path to the saved image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    
    # Generate output path if not provided
    if output_path is None:
        filename = f"heatmap_overlay_{uuid.uuid4().hex[:6]}.png"
        output_path = os.path.join(output_dir, filename)
    
    # Save image
    image.save(output_path)
    
    return output_path


def create_visualization(original_image, heatmap, roi_mask=None, output_dir="reports"):
    """
    Create and save visualization with heatmap overlay.
    
    Args:
        original_image: Original image (PIL Image or numpy array)
        heatmap: Heatmap from Grad-CAM (numpy array)
        roi_mask: Binary mask for the region of interest (optional)
        output_dir: Directory to save the visualization
        
    Returns:
        Dictionary with paths to saved visualizations
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Create heatmap overlay
    overlay = apply_heatmap_with_roi(original_image, heatmap, roi_mask)
    
    # Save visualizations
    overlay_path = save_visualization(overlay, output_dir=output_dir)
    
    # Create result dictionary
    result = {
        "explain_id": os.path.basename(overlay_path).split('.')[0],
        "explain_url": overlay_path,
        "heatmap_overlay_path": overlay_path
    }
    
    return result


def main():
    """
    Main function for testing the module.
    """
    # Example usage
    from PIL import Image
    import numpy as np
    
    # Load sample image
    image_path = "data/test_image.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Create dummy heatmap
        heatmap = np.random.rand(224, 224)
        
        # Create visualization
        result = create_visualization(image, heatmap)
        
        print(f"Visualization created: {result['explain_url']}")
    else:
        print(f"Sample image not found: {image_path}")


if __name__ == "__main__":
    main()