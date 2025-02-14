import os
import sys
import io
import requests
import torch
import numpy as np
from PIL import Image

# Append the root directory (one level up) to sys.path so we can import config.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config

TOGETHER_API_KEY = config.TOGETHER_API_KEY or ""
if not TOGETHER_API_KEY:
    sys.exit("❌ ERROR: No API key found. Check your .env file in the root directory.")

# Optional prefix check (adjust or remove as needed)
expected_prefixes = ["sk-", "tgp_", "a3b"]
if not any(TOGETHER_API_KEY.startswith(p) for p in expected_prefixes):
    sys.exit("❌ ERROR: API key does not appear valid. Check your .env file.")

print("✅ API key loaded successfully from config.py!")

class TogetherImageGenerator:
    CATEGORY = "Together API"

    def __init__(self):
        print("Initializing TogetherImageGenerator node...")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "An astronaut riding a horse on Mars"}),
                "model": ("STRING", {"default": "black-forest-labs/FLUX.1-schnell-Free"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"

    def generate_image(self, prompt, model, width, height, steps):
        """
        1. Uses 'width' and 'height' in the payload to ask the API for that dimension.
        2. Downloads the resulting image, logs shape info for debugging.
        3. (Optional) Resizes the image to (width, height) if the API does not honor the requested dimension.
        4. Returns a PyTorch tensor of shape (H, W, 3) in uint8, so fast image save can do .cpu().numpy().
        """

        print(f"🌀 generate_image called with width={width}, height={height}")

        # 1. Prepare the request
        url = "https://api.together.xyz/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "steps": steps,
            "n": 1,
            "width": width,
            "height": height,
            "guidance": 3.5,
            "output_format": "jpeg"
        }

        try:
            # 2. Send the request
            resp = requests.post(url, json=payload, headers=headers)
            if resp.status_code == 401:
                print("❌ Invalid or missing API key (401). Returning placeholder.")
                return self.placeholder_image(width, height)
            if resp.status_code != 200:
                print(f"❌ API request failed with status {resp.status_code}. Returning placeholder.")
                return self.placeholder_image(width, height)

            data = resp.json()
            if "data" not in data or not data["data"]:
                print("❌ No 'data' field in response. Returning placeholder.")
                return self.placeholder_image(width, height)

            image_url = data["data"][0]["url"]
            print("✅ Image URL from API:", image_url)

            # 3. Download the image
            img_resp = requests.get(image_url)
            if img_resp.status_code != 200:
                print(f"❌ Failed to download image from {image_url}. Returning placeholder.")
                return self.placeholder_image(width, height)

            # Convert to PIL
            pil_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            print("🔍 Original downloaded image size:", pil_img.size)  # (width, height)

            # (Optional) If the API does not produce the exact size, you can resize:
            if pil_img.size != (width, height):
                print(f"⚠️ API returned size {pil_img.size}, but we asked for {(width, height)}. Resizing now...")
                pil_img = pil_img.resize((width, height), Image.LANCZOS)

            # Convert to a float32 array or just do uint8 directly
            # We'll do uint8 so we can pass it directly to the fast image saver
            np_img = np.array(pil_img, dtype=np.uint8)  # shape: (H, W, 3), range 0..255
            print("🔍 Final numpy shape (H, W, C):", np_img.shape)

            # Convert to PyTorch tensor shape (H, W, 3), dtype uint8
            img_tensor = torch.from_numpy(np_img)  # shape: (H, W, 3), dtype=uint8
            print("✅ Final tensor shape:", tuple(img_tensor.shape), "dtype:", img_tensor.dtype)

            return (img_tensor,)

        except Exception as e:
            print("❌ Exception in generate_image:", e)
            return self.placeholder_image(width, height)

    def placeholder_image(self, width, height):
        """
        Creates a red placeholder image of shape (H, W, 3), dtype=uint8
        """
        print("Generating placeholder image for fallback.")
        pil_img = Image.new("RGB", (width, height), color=(255, 0, 0))
        np_img = np.array(pil_img, dtype=np.uint8)  # shape: (H, W, 3)
        return (torch.from_numpy(np_img),)


NODE_CLASS_MAPPINGS = {
    "TogetherImageGenerator": TogetherImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGenerator": "Together Image Generator",
}

print("✅ TogetherImageGenerator node loaded with debug logs for shape checks!")
