import unittest
from nodes.together_image_generator import TogetherImageGenerator

class TestTogetherImageGenerator(unittest.TestCase):
    def setUp(self):
        self.node = TogetherImageGenerator()

    def test_generate_image(self):
        prompt = "A futuristic cityscape with neon lights"
        model = "black-forest-labs/FLUX.1-dev-lora"
        width = 1024
        height = 768
        steps = 28
        lora_urls = "https://example.com/lora1.safetensors, https://example.com/lora2.safetensors"
        lora_scales = "0.8, 1.2"

        output = self.node.generate_image(prompt, model, width, height, steps, lora_urls, lora_scales)
        self.assertIsInstance(output, tuple)

if __name__ == '__main__':
    unittest.main()
