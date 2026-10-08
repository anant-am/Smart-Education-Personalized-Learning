"""
Smart Education Project — Environment and Hardware Validator
=============================================================
Inspects hardware capabilities, Python libraries, CUDA status, disk space,
and dataset/checkpoint paths to ensure the host environment is ready for
development, testing, or full model training.

Usage:
  python scripts/check_environment.py
  python scripts/check_environment.py --require-datasets
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import shutil
import platform
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def format_bytes(b: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if b < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"

def main():
    parser = argparse.ArgumentParser(description="Check environment and hardware readiness")
    parser.add_argument("--require-datasets", action="store_true",
                        help="Enforce that raw EdNet dataset paths exist (fails if missing)")
    args = parser.parse_args()

    print("=" * 80)
    print("  SMART EDUCATION — ENVIRONMENT & HARDWARE INTEGRITY CHECK")
    print("=" * 80)

    all_passed = True
    warnings = []

    # 1. Python Environment
    py_ver = platform.python_version()
    print(f"\n[1] Python Environment:")
    print(f"    - Python Version:       {py_ver}")
    print(f"    - Executable:           {sys.executable}")
    if sys.version_info < (3, 9):
        print("    [!] WARNING: Python 3.9+ is recommended.")
        warnings.append("Python version is below 3.9")

    # 2. PyTorch & GPU Acceleration
    print(f"\n[2] PyTorch & GPU Acceleration:")
    try:
        import torch
        print(f"    - PyTorch Version:      {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"    - CUDA Available:       {cuda_avail}")
        if cuda_avail:
            print(f"    - CUDA Version:         {torch.version.cuda}")
            print(f"    - Device Count:         {torch.cuda.device_count()}")
            print(f"    - Device Name:          {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory
            print(f"    - Total GPU VRAM:       {format_bytes(vram)}")
        else:
            print("    [-] Notice: Running in CPU mode (CUDA not available).")
            print("        For training 298k users, an NVIDIA GPU with CUDA is strongly recommended.")
    except ImportError:
        print("    [!] ERROR: PyTorch is not installed in the current environment.")
        all_passed = False

    # 3. CPU and System Memory
    print(f"\n[3] Host System Hardware:")
    print(f"    - OS:                   {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"    - Processor:            {platform.processor() or 'Standard'}")
    cpu_count = os.cpu_count() or 1
    print(f"    - CPU Logical Cores:    {cpu_count}")

    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"    - Total RAM:            {format_bytes(vm.total)}")
        print(f"    - Available RAM:        {format_bytes(vm.available)}")
    except ImportError:
        print("    - RAM:                  (psutil not installed, skipped)")

    # 4. Storage & Disk Space
    print(f"\n[4] Storage & Disk Capacity:")
    proj_drive = PROJECT_ROOT.anchor if PROJECT_ROOT.anchor else "."
    try:
        disk_usage = shutil.disk_usage(proj_drive)
        print(f"    - Project Drive ({proj_drive}):")
        print(f"        Total:     {format_bytes(disk_usage.total)}")
        print(f"        Used:      {format_bytes(disk_usage.used)}")
        print(f"        Free:      {format_bytes(disk_usage.free)}")
        if disk_usage.free < 5 * 1024**3:
            print("    [!] WARNING: Less than 5 GB free disk space available.")
            warnings.append("Low disk space (< 5 GB)")
    except Exception as e:
        print(f"    - Could not query disk usage: {e}")

    # 5. Project Configuration & Dataset Paths
    print(f"\n[5] Configured Dataset & Checkpoint Paths:")
    try:
        from config import (
            EDNET_KT3_FULL_DIR, EDNET_CONTENTS_DIR,
            PROCESSED_DATA_DIR, CHECKPOINTS_DIR, QUESTIONS_CSV
        )
        print(f"    - Project Root:         {PROJECT_ROOT}")
        print(f"    - Checkpoints Dir:      {CHECKPOINTS_DIR} (Exists: {CHECKPOINTS_DIR.exists()})")
        print(f"    - Processed Data Dir:   {PROCESSED_DATA_DIR} (Exists: {PROCESSED_DATA_DIR.exists()})")
        print(f"    - EdNet-KT3 Root:       {EDNET_KT3_FULL_DIR} (Exists: {EDNET_KT3_FULL_DIR.exists()})")
        print(f"    - EdNet-Contents Dir:   {EDNET_CONTENTS_DIR} (Exists: {EDNET_CONTENTS_DIR.exists()})")
        print(f"    - Questions CSV:        {QUESTIONS_CSV} (Exists: {QUESTIONS_CSV.exists()})")

        if not CHECKPOINTS_DIR.exists():
            print(f"    [!] Checkpoints directory does not exist: {CHECKPOINTS_DIR}")
            all_passed = False

        if args.require_datasets:
            if not EDNET_KT3_FULL_DIR.exists():
                print(f"    [!] REQUIRED DATASET MISSING: EdNet-KT3 directory not found at {EDNET_KT3_FULL_DIR}")
                print(f"        Please set EDNET_KT3_ROOT in .env or copy datasets.")
                all_passed = False
            if not QUESTIONS_CSV.exists():
                print(f"    [!] REQUIRED DATASET MISSING: questions.csv not found at {QUESTIONS_CSV}")
                print(f"        Please set EDNET_CONTENTS_ROOT in .env or copy datasets.")
                all_passed = False
        else:
            if not EDNET_KT3_FULL_DIR.exists() or not QUESTIONS_CSV.exists():
                print("    [-] Notice: Raw EdNet datasets are not yet present at configured paths.")
                print("        (Normal for fresh clean-clone before external datasets are supplied).")
    except Exception as e:
        print(f"    [!] Error importing config: {e}")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("  RESULT: PASS")
        if warnings:
            print(f"  (with {len(warnings)} non-critical warning(s))")
        print("=" * 80)
        sys.exit(0)
    else:
        print("  RESULT: FAIL — Required environment dependencies or paths missing.")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()
