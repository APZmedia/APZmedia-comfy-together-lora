from together import Together
import base64
import io
from PIL import Image
import numpy as np

class TogetherImageGeneratorLoRA:
    def __init__(self):
        self.client = Together()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "An astronaut riding a horse on Mars"}),
                "model": ("STRING", {"default": "black-forest-labs/FLUX.1-dev-lora"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
                "lora_urls": ("STRING", {"default": "", "multiline": True}),
                "lora_scales": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "Together API"

    def generate_image(self, prompt, model, width, height, steps, lora_urls, lora_scales):
        lora_urls_list = [url.strip() for url in lora_urls.split(",") if url.strip()]
        lora_scales_list = [float(scale.strip()) for scale in lora_scales.split(",") if scale.strip()]

        if len(lora_urls_list) != len(lora_scales_list):
            raise ValueError("The number of LoRA URLs must match the number of LoRA scales.")

        image_loras = [{"path": url, "scale": scale} for url, scale in zip(lora_urls_list, lora_scales_list)]

        response = self.client.images.generate(
            prompt=prompt,
            model=model,
            width=width,
            height=height,
            steps=steps,
            n=1,
            response_format="b64_json",
            image_loras=image_loras
        )

        image_data = response.data[0].b64_json
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        img_np = np.array(img) / 255.0

        return (img_np,)

NODE_CLASS_MAPPINGS = {
    "TogetherImageGeneratorLoRA": TogetherImageGeneratorLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGeneratorLoRA": "Together Image Generator with LoRA",
}
