import base64
import io
import os
import sys
import requests
from dotenv import load_dotenv
from PIL import Image
import numpy as np

# ✅ Load .env from the correct location
NODE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current file's directory
ENV_PATH = os.path.join(NODE_DIR, "..", ".env")  # Adjust if needed

print(f"📂 Loading .env from: {ENV_PATH}", flush=True)
load_dotenv(ENV_PATH)

# ✅ Fetch API Key securely & strip whitespace
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()

# ✅ Validate API Key before proceeding
if not TOGETHER_API_KEY:
    print("❌ ERROR: API key is missing or not loaded from .env! Make sure the .env file exists and contains TOGETHER_API_KEY.", flush=True)
    sys.exit(1)

# ✅ Debugging: Print first 5 chars of the API key for verification
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

            if response.status_code != 200:
                print(f"❌ ERROR: API request failed! Status: {response.status_code}")
                print("Response Body:", response.text)
                return None  # ✅ Ensure that an explicit failure response is given

            print("✅ Image generated successfully! Processing output...", flush=True)

            # ✅ Extract image URL from response
            data = response.json()
            if "data" not in data or not data["data"]:
                print("❌ ERROR: API response is missing 'data' field.")
                print("Response Body:", response.text)
                return None

            image_url = data["data"][0]["url"]
            print(f"🔗 Image URL: {image_url}", flush=True)

            # ✅ Download the image
            img_response = requests.get(image_url)
            if img_response.status_code != 200:
                print(f"❌ ERROR: Failed to download image from {image_url}")
                return None

            img = Image.open(io.BytesIO(img_response.content))
            img = img.convert("RGB")
            img_np = np.array(img) / 255.0

            print("✅ Image processing complete! Returning image.", flush=True)
            return (img_np,)

        except Exception as e:
            print(f"❌ API Error: {e}", flush=True)
            return None  # ✅ Explicitly return None in case of error


# ✅ Register the node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "TogetherImageGenerator": TogetherImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGenerator": "Together Image Generator",
}

print("✅ TogetherImageGenerator node successfully loaded using requests!", flush=True)
