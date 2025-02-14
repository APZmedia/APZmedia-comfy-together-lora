from together import Together
import base64
import io
import os
import sys
from dotenv import load_dotenv
from PIL import Image
import numpy as np

# Load .env from the custom node directory
NODE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current file's directory
ENV_PATH = os.path.join(NODE_DIR, "..", ".env")  # Adjust this if needed

print(f"📂 Loading .env from: {ENV_PATH}", flush=True)  # Debug print
load_dotenv(ENV_PATH)

# Fetch API Key securely
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Debugging: Print API Key (Only first 5 chars for security)
if TOGETHER_API_KEY:
    print(f"🔍 Debug: TOGETHER_API_KEY (before initialization) = {TOGETHER_API_KEY[:5]}...", flush=True)
else:
    print("❌ Error: API key is missing or .env file is not loading!", flush=True)
    sys.exit(1)

# Explicitly set API Key for Together API
Together.api_key = TOGETHER_API_KEY  # ✅ Correct API key assignment

class TogetherImageGenerator:
    CATEGORY = "Together API"

    def __init__(self):
        print("🔄 Initializing TogetherImageGenerator node...", flush=True)
        self.client = Together()  # ✅ Corrected client initialization
        print("✅ Together API client initialized successfully!", flush=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "An astronaut riding a horse on Mars"}),
                "model": ("STRING", {"default": "black-forest-labs/FLUX.1-schnell"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)  # ✅ Fix missing return name
    FUNCTION = "generate_image"

    def generate_image(self, prompt, model, width, height, steps):
        print("🔄 Generating image using Together API...", flush=True)
        
        print(f"📤 Debug: API Key being used in request = {TOGETHER_API_KEY[:5]}...", flush=True)
        print("📤 Sending request to Together API...", flush=True)
        try:
            response = self.client.images.generate(
                prompt=prompt,
                model=model,
                width=width,
                height=height,
                steps=steps,
                n=1,
                response_format="b64_json"
            )

            if not hasattr(response, "data") or not response.data:
                raise ValueError("❌ Error: API response is empty or incorrect!")

            print("✅ Image generated successfully! Processing output...", flush=True)

            image_data = response.data[0].b64_json
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert("RGB")
            img_np = np.array(img) / 255.0
            print("✅ Image processing complete! Returning image.", flush=True)

            return (img_np,)

        except Exception as e:
            print(f"❌ API Error: {e}", flush=True)
            raise

NODE_CLASS_MAPPINGS = {
    "TogetherImageGenerator": TogetherImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGenerator": "Together Image Generator",
}

print("✅ TogetherImageGenerator node successfully loaded!", flush=True)
