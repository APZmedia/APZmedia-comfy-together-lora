import together
import os
from dotenv import load_dotenv

# Load the .env file
NODE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current file's directory
ENV_PATH = os.path.join(NODE_DIR, "..", ".env")  # Adjust if needed
load_dotenv(ENV_PATH)

# Fetch API Key securely
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

print(f"🔍 Testing API Key: {TOGETHER_API_KEY[:5]}...")

try:
    client = together.Together(api_key=TOGETHER_API_KEY)
    response = client.images.generate(
        prompt="A futuristic cityscape",
        model="black-forest-labs/FLUX.1-schnell",
        width=1024,
        height=768,
        steps=30,
        n=1,
        response_format="b64_json"
    )
    print("✅ API Key Works! 🎉")
except together.error.AuthenticationError as e:
    print(f"❌ API Key Error: {e}")
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
