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

# Debugging: Print API Key
if TOGETHER_API_KEY:
    print(f"🔍 Debug: TOGETHER_API_KEY (before initialization) = {TOGETHER_API_KEY}", flush=True)
else:
    print("❌ Error: API key is missing or .env file is not loading!", flush=True)
    sys.exit(1)

class TogetherImageGenerator:
    def __init__(self):
        print("🔄 Initializing TogetherImageGenerator node...", flush=True)

        # ✅ Explicitly set API key when creating client
        self.client = Together(api_key=TOGETHER_API_KEY)  

        print("✅ Together API client initialized successfully!", flush=True)

    def generate_image(self, prompt, model, width, height, steps):
        print("🔄 Generating image using Together API...", flush=True)

        print(f"📤 Debug: API Key being used in request = {TOGETHER_API_KEY}", flush=True)

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
