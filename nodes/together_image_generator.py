import os
import sys
import base64
import io
import requests
import torch  # Import PyTorch
from PIL import Image
import numpy as np

# Append the root directory (one level up) to sys.path so we can import config
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Now import the config module
import config

# Use the API key from config.py
TOGETHER_API_KEY = config.TOGETHER_API_KEY
if not TOGETHER_API_KEY:
    print("❌ ERROR: API key is missing in config.py! Check your .env file in the root directory.", flush=True)
    sys.exit(1)
print(f"🔍 Debug: TOGETHER_API_KEY (first 5 chars) = {TOGETHER_API_KEY[:5]}...", flush=True)


class TogetherImageGenerator:
    CATEGORY = "Together API"

    def __init__(self):
        print("🔄 Initializing TogetherImageGenerator node...", flush=True)
        print("✅ API will use direct requests method!", flush=True)

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
        print("🔄 Generating image using Together API...", flush=True)

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
            print("🔄 Full API Response:", response.text)  # Log the full API response

            if response.status_code == 401:
                print("❌ ERROR: Invalid API Key! Make sure it is correct.", flush=True)
                return self.placeholder_image(width, height)
            if response.status_code != 200:
                print(f"❌ ERROR: API request failed! Status: {response.status_code}", flush=True)
                return self.placeholder_image(width, height)

            print("✅ Image generated successfully! Processing output...", flush=True)
            data = response.json()
            if "data" not in data or not data["data"]:
                print("❌ ERROR: API response is missing 'data' field.", flush=True)
                return self.placeholder_image(width, height)

            image_url = data["data"][0]["url"]
            print(f"🔗 Image URL: {image_url}", flush=True)

            img_response = requests.get(image_url)
            if img_response.status_code != 200:
                print(f"❌ ERROR: Failed to download image from {image_url}", flush=True)
                return self.placeholder_image(width, height)

            img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
            img_np = np.array(img, dtype=np.float32) / 255.0
            if img_np.ndim == 2:  # if grayscale, convert to RGB
                img_np = np.stack([img_np] * 3, axis=-1)
            img_np = np.moveaxis(img_np, -1, 0)  # (H, W, 3) -> (3, H, W)
            img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)  # [1, 3, H, W]

            print("✅ Image processing complete! Returning image.", flush=True)
            return (img_tensor,)

        except Exception as e:
            print(f"❌ API Error: {e}", flush=True)
            return self.placeholder_image(width, height)

    def placeholder_image(self, width, height):
        """
        Generate a blank placeholder image with a red background.
        """
        img = Image.new("RGB", (width, height), color=(255, 0, 0))
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.moveaxis(img_np, -1, 0)  # (H, W, 3) -> (3, H, W)
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
        return (img_tensor,)


# Register the node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "TogetherImageGenerator": TogetherImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGenerator": "Together Image Generator",
}

print("✅ TogetherImageGenerator node successfully loaded using requests!", flush=True)
