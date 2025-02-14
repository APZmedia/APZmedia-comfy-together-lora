import os
import sys
import io
import requests
import torch  # Import PyTorch
from PIL import Image
import numpy as np

# Append the root directory (one level up) to sys.path so we can import config.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
import importlib
importlib.reload(config)

TOGETHER_API_KEY = config.TOGETHER_API_KEY
if not TOGETHER_API_KEY:
    print("❌ ERROR: API key is missing in config.py! Check your .env file in the root directory.")
    # Instead of exiting, we continue and always return a placeholder image.
else:
    print("API key loaded successfully from config.py!")

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
        # If no API key is provided, return a placeholder immediately.
        if not TOGETHER_API_KEY:
            print("❌ No valid API key provided. Returning placeholder image.")
            return self.placeholder_image(width, height)

        print(f"Generating image using Together API with requested dimensions ({width} x {height})...")
        
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
                print("❌ ERROR: Invalid API Key! Aborting API call and returning placeholder image.")
                return self.placeholder_image(width, height)
            if resp.status_code != 200:
                print(f"❌ ERROR: API request failed with status {resp.status_code}. Returning placeholder image.")
                return self.placeholder_image(width, height)
            
            data = resp.json()
            if "data" not in data or not data["data"]:
                print("❌ ERROR: API response is missing 'data' field. Returning placeholder image.")
                return self.placeholder_image(width, height)
            
            image_url = data["data"][0]["url"]
            print("Image URL from API:", image_url)
            
            img_resp = requests.get(image_url)
            if img_resp.status_code != 200:
                print(f"❌ ERROR: Failed to download image from {image_url}. Returning placeholder image.")
                return self.placeholder_image(width, height)
            
            pil_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            print("Original downloaded image size:", pil_img.size)
            
            # Resize if necessary (optional)
            if pil_img.size != (width, height):
                print(f"⚠️ Resizing image from {pil_img.size} to ({width}, {height})...")
                pil_img = pil_img.resize((width, height), Image.LANCZOS)
            
            # Convert image to a NumPy array with dtype float32, values in [0, 1]
            np_img = np.array(pil_img, dtype=np.float32) / 255.0  # shape: (H, W, 3)
            print("Final numpy shape (H, W, C):", np_img.shape)
            
            # Return as a PyTorch tensor in float32 with shape (H, W, 3)
            img_tensor = torch.from_numpy(np_img)  # dtype=torch.float32, shape: (H, W, 3)
            print("Final tensor shape:", tuple(img_tensor.shape), "dtype:", img_tensor.dtype)
            
            return (img_tensor,)
        
        except Exception as e:
            print("❌ Exception in generate_image:", e)
            return self.placeholder_image(width, height)
    
    def placeholder_image(self, width, height):
        print("Generating placeholder image...")
        pil_img = Image.new("RGB", (width, height), color=(255, 0, 0))
        np_img = np.array(pil_img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(np_img)
        return (img_tensor,)

NODE_CLASS_MAPPINGS = {
    "TogetherImageGenerator": TogetherImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGenerator": "Together Image Generator",
}

print("TogetherImageGenerator node successfully loaded!")
