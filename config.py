import os
from dotenv import load_dotenv

# Force reload .env even if variables are already set
load_dotenv(override=True)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()
print("DEBUG: Loaded API key from .env:", TOGETHER_API_KEY)  # for debugging
