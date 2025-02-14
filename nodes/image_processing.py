import io
import requests
import torch
import numpy as np
from PIL import Image, ImageOps

def download_and_process_image(image_url, width, height):
    """
    Downloads an image from the provided URL, ensures proper orientation and RGB format,
    and returns a 4D PyTorch tensor shaped (1, h, w, 3), normalized to [0,1].
    """
    if not image_url:
        print("❌ ERROR: No image URL provided.")
        return placeholder_image(width, height)

    try:
        resp = requests.get(image_url, timeout=10)
        if resp.status_code != 200:
            print(f"❌ ERROR: Failed to download image from {image_url}")
            return placeholder_image(width, height)

        # 1. Open as a PIL image
        pil_img = Image.open(io.BytesIO(resp.content))

        # 2. Fix orientation (physically rotates pixels if needed)
        pil_img = ImageOps.exif_transpose(pil_img)

        # 3. Convert to RGB
        pil_img = pil_img.convert("RGB")

        # 4. Convert to NumPy (h, w, 3), normalized to [0,1]
        np_img = np.array(pil_img, dtype=np.float32) / 255.0

        # 5. Ensure shape is (h, w, 3)
        if np_img.ndim == 2:
            # Grayscale => Expand/repeat to get 3 identical channels
            np_img = np.expand_dims(np_img, axis=-1)  # (h, w, 1)
            np_img = np.repeat(np_img, 3, axis=-1)    # (h, w, 3)
        elif np_img.shape[-1] != 3:
            # If there's an alpha channel or something else, just keep first 3
            np_img = np_img[..., :3]

        # (Optional) Resize if you MUST enforce exact (width, height).
        # Otherwise, skip or do 'pil_img.resize((width, height))' above if needed.
        # For example:
        # if (np_img.shape[1], np_img.shape[0]) != (width, height):
        #     pil_img = pil_img.resize((width, height), Image.Resampling.LANCZOS)
        #     np_img = np.array(pil_img, dtype=np.float32) / 255.0

        # 6. Convert to torch, shape => (h, w, 3)
        img_tensor = torch.from_numpy(np_img).contiguous()

        # 7. Add batch dimension => (1, h, w, 3)
        img_tensor = img_tensor.unsqueeze(0)

        print(f"✅ Final tensor shape: {tuple(img_tensor.shape)} (should be (1, h, w, 3))")
        return img_tensor

    except Exception as e:
        print(f"❌ ERROR: Exception in image processing: {e}")
        return placeholder_image(width, height)

def placeholder_image(width, height):
    """
    Generates a red placeholder image.
    Returns a 4D torch tensor of shape (1, h, w, 3) with values in [0,1].
    """
    print("⚠️ Generating placeholder image...")
    pil_img = Image.new("RGB", (width, height), color=(255, 0, 0))
    np_img = np.array(pil_img, dtype=np.float32) / 255.0  # => (h, w, 3)

    img_tensor = torch.from_numpy(np_img).contiguous()    # => (h, w, 3)
    img_tensor = img_tensor.unsqueeze(0)                  # => (1, h, w, 3)
    return img_tensor
