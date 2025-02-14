import os
from dotenv import load_dotenv

load_dotenv(".env")  # Adjust path as needed
value = os.getenv("TOGETHER_API_KEY")
print(f"My Together key is: {value}")
