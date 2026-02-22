import os
import uuid
import numpy as np
from PIL import Image
from .gradcam import GradCAM

def process_image_with_gradcam(model, target_layer, image_tensor, original_image):
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(image_tensor, class_idx=None)
    heatmap = (cam * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).convert("RGBA")
    base_img = Image.fromarray(original_image).convert("RGBA")
    heatmap_colored = Image.new("RGBA", heatmap_img.size)
    r = heatmap_img.split()[0]
    heatmap_colored.putalpha(r)
    overlay = Image.alpha_composite(base_img.resize(heatmap_img.size), heatmap_colored)
    os.makedirs("reports", exist_ok=True)
    explain_id = str(uuid.uuid4())
    out_path = os.path.join("reports", f"gradcam_{explain_id}.png")
    overlay.convert("RGB").save(out_path)
    return {
        "explain_id": explain_id,
        "explain_url": out_path,
        "heatmap_overlay_path": out_path,
    }