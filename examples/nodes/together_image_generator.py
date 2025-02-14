from together import Together
import base64
import io
from PIL import Image
import numpy as np

class TogetherImageGenerator:
    def __init__(self):
        print("🔄 Initializing TogetherImageGenerator node...")
        self.client = Together()
        print("✅ Together API client initialized successfully!")

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
        print("🔄 Generating image using Together API...")
        
        print("📤 Sending request to Together API...")
        response = self.client.images.generate(
            prompt=prompt,
            model=model,
            width=width,
            height=height,
            steps=steps,
            n=1,
            response_format="b64_json"
        )

        print("✅ Image generated successfully! Processing output...")
        image_data = response.data[0].b64_json
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        img_np = np.array(img) / 255.0
        print("✅ Image processing complete! Returning image.")

        return (img_np,)

NODE_CLASS_MAPPINGS = {
    "TogetherImageGenerator": TogetherImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGenerator": "Together Image Generator",
}

print("✅ TogetherImageGenerator node successfully loaded!")
