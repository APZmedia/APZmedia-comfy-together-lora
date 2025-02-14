import os
import sys
import base64
import io
import requests
import torch  # Import PyTorch
from PIL import Image
import numpy as np

# Append the root directory (one level up) to sys.path to import config.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config

# Use the API key from config.py
TOGETHER_API_KEY = config.TOGETHER_API_KEY
if not TOGETHER_API_KEY:
    sys.exit("❌ ERROR: API key is missing in config.py! Check your .env file in the root directory.")
print("API key loaded successfully.")

class TogetherImageGenerator:
    CATEGORY = "Together API"

    def __init__(self):
        print("Initializing TogetherImageGenerator node...")
        # Use the API key from the imported config
        # (If you prefer, you can pass it explicitly to your API client here.)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "An astronaut riding a horse on Mars"}),
                "model": ("STRING", {"default": "black-forest-labs/FLUX.1-schnell-Free"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"

    def generate_image(self, prompt, model, width, height, steps):
        print("Generating image using Together API...")

        url = "https://api.together.xyz/v1/images/generations"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {TOGETHER_API_KEY}",
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "steps": steps,
            "n": 1,
            "height": height,
            "width": width,
            "guidance": 3.5,
            "output_format": "jpeg"
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                print("API request failed:", response.text)
                return self.placeholder_image(width, height)

            data = response.json()
            if "data" not in data or not data["data"]:
                print("API response missing 'data' field.")
                return self.placeholder_image(width, height)

            image_url = data["data"][0]["url"]
            print("Image URL:", image_url)

            img_response = requests.get(image_url)
            if img_response.status_code != 200:
                print("Failed to download image from", image_url)
                return self.placeholder_image(width, height)

            img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
            # Process image: convert to NumPy array (H, W, 3) then to uint8 format
            img_np = np.array(img, dtype=np.float32) / 255.0
            img_np_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
            print("Image processed successfully.")
            return (img_np_uint8,)

        except Exception as e:
            print("API Error:", e)
            return self.placeholder_image(width, height)

    def placeholder_image(self, width, height):
        # Create a red placeholder image (H, W, 3) in uint8 format
        img = Image.new("RGB", (width, height), color=(255, 0, 0))
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        return (img_np_uint8,)

# Register the node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "TogetherImageGenerator": TogetherImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGenerator": "Together Image Generator",
}

print("TogetherImageGenerator node successfully loaded!")
