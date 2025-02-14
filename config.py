import os
from dotenv import load_dotenv

# Load the .env file from the root directory
load_dotenv()

# Fetch and store the Together API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()
