from .nodes.together_image_generator import TogetherImageGenerator
from .nodes.together_image_generator_lora import TogetherImageGeneratorLoRA

NODE_CLASS_MAPPINGS = {
    "TogetherImageGenerator": TogetherImageGenerator,
    "TogetherImageGeneratorLoRA": TogetherImageGeneratorLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGenerator": "Together Image Generator",
    "TogetherImageGeneratorLoRA": "Together Image Generator with LoRA",
}

print("✅ APZmedia Comfy-Together-Lora nodes successfully loaded!")
