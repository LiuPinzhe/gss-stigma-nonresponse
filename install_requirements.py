#!/usr/bin/env python3
"""
Install additional requirements for ASL integration
"""
import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")

if __name__ == "__main__":
    packages = ["torch", "torchvision"]
    
    print("Installing PyTorch for ASL loss function...")
    for package in packages:
        install_package(package)
    
    print("\nInstallation complete! You can now run:")
    print("python gss_stigma_asl.py --data data/GSS.xlsx --out outputs_asl --mode composite")