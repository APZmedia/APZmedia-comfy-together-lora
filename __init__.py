import os
import sys
import importlib
from dotenv import load_dotenv

# Get the absolute path to the custom nodes directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR))

# Ensure the root directory is in sys.path to allow config.py imports
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Load .env file manually
env_loaded = load_dotenv(os.path.join(ROOT_DIR, ".env"))
if not env_loaded:
    print("⚠️ WARNING: .env file not found or could not be loaded!")

# Try to import config.py dynamically
try:
    config_path = os.path.join(ROOT_DIR, "config.py")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ ERROR: config.py not found in {ROOT_DIR}")

    # Import config as a module
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    TOGETHER_API_KEY = getattr(config, "TOGETHER_API_KEY", None)

    if not TOGETHER_API_KEY:
        print("❌ ERROR: TOGETHER_API_KEY is missing or empty in config.py!")

except Exception as e:
    print(f"❌ ERROR loading config.py: {e}")
    TOGETHER_API_KEY = None

# Import nodes AFTER loading config
from .nodes.together_image_generator import TogetherImageGenerator
from .nodes.together_image_generator_lora import TogetherImageGeneratorLoRA

# Inject API key into nodes
TogetherImageGenerator.TOGETHER_API_KEY = TOGETHER_API_KEY
TogetherImageGeneratorLoRA.TOGETHER_API_KEY = TOGETHER_API_KEY

# Register node mappings
NODE_CLASS_MAPPINGS = {
    "TogetherImageGenerator": TogetherImageGenerator,
    "TogetherImageGeneratorLoRA": TogetherImageGeneratorLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherImageGenerator": "Together Image Generator",
    "TogetherImageGeneratorLoRA": "Together Image Generator with LoRA",
}

# ✅ Fixed the unterminated string
if TOGETHER_API_KEY:
    print(f"✅ APZmedia Comfy-Together-Lora nodes loaded with API key: {TOGETHER_API_KEY[:5]}...")
else:
    print("❌ API key not found!")
