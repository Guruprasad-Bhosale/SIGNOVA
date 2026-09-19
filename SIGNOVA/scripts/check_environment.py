#!/usr/bin/env python3
"""
SIGNOVA Environment Checker
Verifies host system capabilities, Python version, hardware acceleration (NVIDIA CUDA / GPU),
and key computer vision / machine learning libraries.
"""

import os
import platform
import shutil
import sys
from pathlib import Path


def get_disk_space(path: str = ".") -> str:
    """Return free and total disk space in human-readable GB."""
    try:
        total, used, free = shutil.disk_usage(path)
        return f"{free / (1024**3):.2f} GB free / {total / (1024**3):.2f} GB total"
    except Exception:
        return "Unknown"


def get_ram_info() -> str:
    """Return available system RAM."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return f"{mem.total / (1024**3):.2f} GB (Available: {mem.available / (1024**3):.2f} GB)"
    except ImportError:
        # Fallback for Windows without psutil
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return f"{stat.ullTotalPhys / (1024**3):.2f} GB (Available: {stat.ullAvailPhys / (1024**3):.2f} GB)"
        except Exception:
            return "Unable to query (psutil not installed)"


def get_gpu_info():
    """Detect NVIDIA GPU and CUDA driver capabilities."""
    gpu_info = {"detected": False, "name": "None", "vram": "N/A", "driver": "N/A"}
    
    # Try pycuda or torch first
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["detected"] = True
            gpu_info["name"] = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info["vram"] = f"{total_vram:.2f} GB"
            gpu_info["torch_cuda"] = torch.version.cuda
            return gpu_info
    except ImportError:
        pass

    # Fallback to nvidia-smi
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = res.stdout.strip().split(",")
            gpu_info["detected"] = True
            gpu_info["name"] = parts[0].strip() if len(parts) > 0 else "NVIDIA GPU"
            gpu_info["vram"] = parts[1].strip() if len(parts) > 1 else "Unknown"
            gpu_info["driver"] = parts[2].strip() if len(parts) > 2 else "Unknown"
    except Exception:
        pass

    return gpu_info


def check_library(module_name: str, attr_name: str = "__version__") -> str:
    """Safely check if a Python module is installed and return its version."""
    try:
        mod = __import__(module_name)
        return getattr(mod, attr_name, "installed (version unknown)")
    except ImportError:
        return "NOT INSTALLED"


def main():
    print("=" * 60)
    print("           SIGNOVA ENVIRONMENT & HARDWARE CHECK           ")
    print("=" * 60)

    # Python & OS
    print(f"Python Version       : {sys.version.split()[0]} ({sys.executable})")
    print(f"Operating System     : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"CPU Architecture     : {platform.processor() or platform.machine()}")
    print(f"System RAM           : {get_ram_info()}")
    print(f"Disk Free Space      : {get_disk_space()}")

    print("-" * 60)
    print("HARDWARE ACCELERATION (GPU)")
    print("-" * 60)
    gpu = get_gpu_info()
    print(f"GPU Detected         : {gpu.get('name')}")
    print(f"GPU VRAM             : {gpu.get('vram')}")
    if "driver" in gpu and gpu["driver"] != "N/A":
        print(f"NVIDIA Driver        : {gpu.get('driver')}")

    print("-" * 60)
    print("MACHINE LEARNING & COMPUTER VISION LIBRARIES")
    print("-" * 60)

    torch_ver = check_library("torch")
    print(f"PyTorch              : {torch_ver}")
    
    cuda_avail = False
    if torch_ver != "NOT INSTALLED":
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"PyTorch CUDA Active  : {cuda_avail}")
        if cuda_avail:
            print(f"PyTorch CUDA Version : {torch.version.cuda}")
    else:
        print(f"PyTorch CUDA Active  : False (PyTorch not installed)")

    print(f"NumPy                : {check_library('numpy')}")
    print(f"OpenCV (cv2)         : {check_library('cv2')}")
    print(f"MediaPipe            : {check_library('mediapipe')}")
    print(f"FastAPI              : {check_library('fastapi')}")
    print(f"Uvicorn              : {check_library('uvicorn')}")
    print(f"PyYAML               : {check_library('yaml')}")
    print(f"Pydantic             : {check_library('pydantic')}")
    print(f"PyTest               : {check_library('pytest')}")

    print("=" * 60)
    print("Summary:")
    if gpu.get("detected"):
        print(f"  [OK] NVIDIA GPU detected: {gpu.get('name')}")
        if not cuda_avail:
            print("  [NOTE] PyTorch with CUDA is not yet installed in this environment.")
            print("         Install command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("  [INFO] No GPU detected. Running in CPU-only mode.")

    print("=" * 60)


if __name__ == "__main__":
    main()
