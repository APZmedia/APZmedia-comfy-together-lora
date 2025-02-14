from setuptools import setup, find_packages
import subprocess
import sys

def install_dependencies():
    required_packages = [
        "together",
        "numpy",
        "Pillow",
        "python-dotenv"  # If you're using environment variables
    ]
    print("🔄 Checking and installing missing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *required_packages])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

install_dependencies()  # Run auto-install on setup