import requests
from config import TOGETHER_API_KEY

def fetch_image_from_together(prompt, model, width, height, steps):
    """
    Sends a request to Together API to generate an image.
    Returns the image URL if successful, or None if an error occurs.
    """
    if not TOGETHER_API_KEY:
        print("❌ ERROR: Missing API key. Cannot proceed with API call.")
        return None

    url = "https://api.together.xyz/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "steps": steps,
        "n": 1,
        "width": width,
        "height": height,
        "guidance": 3.5,
        "output_format": "jpeg"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"🔄 Sending request to Together API... Status: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ ERROR: API request failed with status {response.status_code}")
            return None

        data = response.json()
        if "data" not in data or not data["data"]:
            print("❌ ERROR: API response missing 'data' field")
            return None

        return data["data"][0]["url"]

    except Exception as e:
        print(f"❌ ERROR: Exception during API request: {e}")
        return None
