import os
import torch
import platform

def cpu_check(platform):
    if "arm64" in platform:
        print(f"Using Python for Arm64: {platform}")
    else:
        print(f"Using Python for x86: {platform}")
    cpu_count = os.cpu_count()
    print(f"CPU has {cpu_count} cores")

def gpu_check():
    if torch.backends.cuda.is_built():
        if torch.backends.cudnn.is_available():
            print(f"Pytorch has been installed with CUDA enabled and CUDNN is available.\tCUDNN enabled:{torch.backends.cudnn.enabled}")
        else:
            print("Pytorch has been installed with CUDA enabled and CUDNN is not available.")
    elif torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("The macOS device supports MPS and Pytorch is built with MPS support")
            mps_device = torch.device("mps")
        else:
            print("The macOS device supports MPS but Pytorch is not built with MPS support")
    else:
        print("Pytorch does not have GPU support for CUDA or Metal Performance Shaders")

# Setup Check
def machine_check_setup():
    plt = platform.platform()
    cpu_check(plt)
    gpu_check()

if __name__ == "__main__":
    machine_check_setup()