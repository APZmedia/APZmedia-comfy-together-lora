# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env (or the actual environment if set)
# Note: You can specify a path to .env if needed:
# load_dotenv(dotenv_path="path/to/.env", override=True)

load_dotenv(override=True)

# Retrieve the Together API key from environment
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()

# Optional: Print for debugging
if TOGETHER_API_KEY:
    print(f"DEBUG: Successfully loaded TOGETHER_API_KEY. Starts with: {TOGETHER_API_KEY[:5]}...")
else:
    print("⚠️ WARNING: TOGETHER_API_KEY is missing or empty!")
