import os
import sys
import io
import requests
import torch
import numpy as np
from PIL import Image

# Ensure we can import config.py from the root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config

# Read the key from config.py
TOGETHER_API_KEY = config.TOGETHER_API_KEY
if not TOGETHER_API_KEY:
    print("❌ No API key found in config.py. Will always return a placeholder image.")
else:
    print("✅ API key loaded from config.py!")

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
        - If we have a valid API key, calls the Together API to generate an image.
        - If the key is invalid or the request fails, returns a placeholder.
        - The final image is returned as a PyTorch float32 tensor in shape (H, W, 3), range [0..1].
        """
        # If no valid key, skip the API call and return placeholder
        if not TOGETHER_API_KEY:
            print("❌ No valid API key. Returning placeholder.")
            return self.placeholder_image(width, height)

        print(f"🚀 Generating image: prompt='{prompt}', size=({width}, {height})")

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
            resp = requests.post(url, json=payload, headers=headers)
            print("Full API Response:", resp.text)

            if resp.status_code == 401:
                print("❌ ERROR: Invalid API Key (401). Returning placeholder.")
                return self.placeholder_image(width, height)
            if resp.status_code != 200:
                print(f"❌ ERROR: API request failed (status {resp.status_code}). Returning placeholder.")
                return self.placeholder_image(width, height)

            data = resp.json()
            if "data" not in data or not data["data"]:
                print("❌ ERROR: No 'data' field in API response. Returning placeholder.")
                return self.placeholder_image(width, height)

            # Extract image URL from response
            image_url = data["data"][0]["url"]
            print("✅ Image URL from Together API:", image_url)

            # Download the image
            img_resp = requests.get(image_url)
            if img_resp.status_code != 200:
                print("❌ ERROR: Failed to download image from URL. Returning placeholder.")
                return self.placeholder_image(width, height)

            # Convert to PIL, ensure RGB
            pil_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            print("Original downloaded image size:", pil_img.size)

            # If you want to force the final image to match (width, height), do:
            if pil_img.size != (width, height):
                print(f"⚠️ Resizing from {pil_img.size} to ({width}, {height})...")
                pil_img = pil_img.resize((width, height), Image.LANCZOS)

            # Convert to float32 [0..1], shape (H, W, 3)
            np_img = np.array(pil_img, dtype=np.float32) / 255.0
            print("Final numpy shape:", np_img.shape)

            # Return as a PyTorch float32 tensor in shape (H, W, 3)
            img_tensor = torch.from_numpy(np_img)
            print("Final tensor shape:", tuple(img_tensor.shape), "dtype:", img_tensor.dtype)

            # Return as a 1-element tuple
            return (img_tensor,)

        except Exception as e:
            print("❌ Exception while generating image:", e)
            return self.placeholder_image(width, height)

    def placeholder_image(self, width, height):
        """
        Returns a red placeholder image in shape (H, W, 3), float32 [0..1].
        """
        print("Generating placeholder image.")
        pil_img = Image.new("RGB", (width, height), (255, 0, 0))
        np_img = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3), float32
        return (torch.from_numpy(np_img),)


NODE_CLASS_MAPPINGS = {
    "TogetherImageGenerator": TogetherImageGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGenerator": "Together Image Generator",
}

print("✅ TogetherImageGenerator node loaded—returns (H, W, 3) float in [0..1]!")
