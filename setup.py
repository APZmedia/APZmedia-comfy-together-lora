from setuptools import setup, find_packages

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
    author="Pablo Apiolazza",
    author_email="me@apzmedia.com",
    url="https://github.com/APZmedia/APZmedia-comfy-together-lora",
)
