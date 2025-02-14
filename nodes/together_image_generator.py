from together import Together
import base64
import io
import os
import sys
from dotenv import load_dotenv
from PIL import Image
import numpy as np

# Load environment variables with explicit path
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# Fetch API Key securely
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Debugging: Print the API key (only first 5 chars for security)
if TOGETHER_API_KEY:
    print(f"🔍 Debug: TOGETHER_API_KEY = {TOGETHER_API_KEY[:5]}...", flush=True)
else:
    print("❌ Error: API key is missing or .env file is not loading!", flush=True)
    sys.exit(1)

# Set API Key in the Together client
Together.api_key = TOGETHER_API_KEY  # ✅ Correct API key assignment

class TogetherImageGenerator:
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
    FUNCTION = "generate_image"
    CATEGORY = "Together API"

    def generate_image(self, prompt, model, width, height, steps):
        print("🔄 Generating image using Together API...", flush=True)
        
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
