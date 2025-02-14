import os
import sys
from .together_api import fetch_image_from_together
from .image_processing import download_and_process_image

# Ensure we can import config.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
import importlib
importlib.reload(config)

TOGETHER_API_KEY = config.TOGETHER_API_KEY
if not TOGETHER_API_KEY:
    print("❌ ERROR: API key is missing in config.py!")
else:
    print("✅ API key loaded successfully!")

class TogetherImageGenerator:
    CATEGORY = "Together API"

    def __init__(self):
        print("🛠️ Initializing TogetherImageGenerator node...")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "An astronaut riding a horse on Mars"}),
                "model": ("STRING", {"default": "black-forest-labs/FLUX.1-schnell-Free"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 1440, "step": 64}),
                "height": ("INT", {"default": 768, "min": 256, "max": 1400, "step": 64}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"

    def generate_image(self, prompt, model, width, height, steps):
        """
        Calls Together API and processes the generated image.
        """
        image_url = fetch_image_from_together(prompt, model, width, height, steps)
        return (download_and_process_image(image_url, width, height),)

NODE_CLASS_MAPPINGS = {
    "TogetherImageGenerator": TogetherImageGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGenerator": "Together Image Generator",
}

print("✅ TogetherImageGenerator node successfully loaded!")
