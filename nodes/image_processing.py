import io
import requests
import torch
import numpy as np
from PIL import Image, ImageOps

def download_and_process_image(image_url, width, height):
    """
    Downloads an image from the provided URL, processes it to ensure:
    - Correct RGB format
    - Orientation fix (EXIF metadata)
    - Converts to a PyTorch tensor (C, H, W)
    """
    if not image_url:
        print("❌ ERROR: No image URL provided.")
        return placeholder_image(width, height)

    try:
        img_resp = requests.get(image_url)
        if img_resp.status_code != 200:
            print(f"❌ ERROR: Failed to download image from {image_url}")
            return placeholder_image(width, height)

        # Open image and convert to RGB
        pil_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")

        # Fix orientation based on EXIF metadata
        pil_img = ImageOps.exif_transpose(pil_img)

        print(f"✅ Image downloaded successfully. Size: {pil_img.size}")

        # Convert to NumPy (H, W, 3)
        np_img = np.array(pil_img, dtype=np.float32) / 255.0
        print(f"🔍 NumPy Image Shape: {np_img.shape}")

        # Convert to PyTorch tensor (C, H, W)
        img_tensor = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()

        print(f"🎯 Final PyTorch Tensor Shape: {img_tensor.shape}")
        return img_tensor

    except Exception as e:
        print(f"❌ ERROR: Exception in image processing: {e}")
        return placeholder_image(width, height)

def placeholder_image(width, height):
    """
    Generates a red placeholder image.
    Returns a PyTorch tensor (C, H, W) with values in [0,1].
    """
    print("⚠️ Generating placeholder image...")
    pil_img = Image.new("RGB", (width, height), color=(255, 0, 0))
    np_img = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
