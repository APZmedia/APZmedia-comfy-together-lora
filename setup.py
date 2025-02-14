from setuptools import setup, find_packages
import subprocess
import sys

def install_dependencies():
    required_packages = [
        "together",
        "numpy",
        "Pillow",
        "python-dotenv"
    ]
    print("🔄 Checking and installing missing dependencies...", flush=True)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *required_packages])
        print("✅ Dependencies installed successfully!", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}", flush=True)
        sys.exit(1)

install_dependencies()  # Run auto-install on setup

setup(
    name="ComfyUI-Together",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "together",
        "numpy",
        "Pillow",
        "python-dotenv"
    ],
    include_package_data=True,
    description="ComfyUI custom node for Together AI image generation",
    author="Your Name",
    author_email="me@apzmedia.com",
    url="https://github.com/APZmedia/APZmedia-comfy-together-lora",
)
