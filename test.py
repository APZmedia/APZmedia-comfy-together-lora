import requests
import os
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Get API Key from .env
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# ✅ Debug: Print API key (only first 5 chars for security)
if TOGETHER_API_KEY:
    print(f"🔍 Loaded API Key: {TOGETHER_API_KEY[:5]}... ✅")
else:
    print("❌ ERROR: API key not found in .env file!")
    exit()

# ✅ API Endpoint
url = "https://api.together.xyz/v1/images/generations"

# ✅ Payload with prompt
payload = {
    "model": "black-forest-labs/FLUX.1-schnell-Free",
    "prompt": "A futuristic cityscape with neon lights",
    "steps": 4,
    "n": 1,
    "height": 1024,
    "width": 1024,
    "guidance": 3.5,
    "output_format": "jpeg"
}

# ✅ Headers with API Key
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {TOGETHER_API_KEY}"  # ✅ API key loaded from .env
}

# ✅ Send the request
response = requests.post(url, json=payload, headers=headers)

# ✅ Print response
print("🔄 Sending request to Together API...")
print(f"Response Code: {response.status_code}")
print("Response Body:")
print(response.text)

# ✅ Handle errors
if response.status_code == 401:
    print("❌ ERROR: Invalid API Key. Check your Together API key and try again.")
elif response.status_code == 400:
    print("❌ ERROR: Bad request. Check required parameters.")
elif response.status_code == 404:
    print("❌ ERROR: Endpoint not found. Check the URL.")
elif response.status_code == 200:
    print("✅ SUCCESS: Image generated!")
