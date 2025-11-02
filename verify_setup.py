#!/usr/bin/env python3
"""
Verify that the environment is set up correctly for DSEC preprocessing.
Run this before attempting to process the full dataset.
"""
import sys
import subprocess
from pathlib import Path

def check_item(description, check_func):
    """Run a check and print result."""
    try:
        result, message = check_func()
        status = "✓" if result else "✗"
        color = "\033[92m" if result else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {description}")
        if message:
            print(f"  → {message}")
        return result
    except Exception as e:
        print(f"\033[91m✗\033[0m {description}")
        print(f"  → Error: {e}")
        return False

def check_python_version():
    """Check Python version >= 3.8."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (need >= 3.8)"

def check_module(module_name):
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        return True, f"{module_name} installed"
    except ImportError:
        return False, f"{module_name} not found - install with: pip install {module_name}"

def check_torch():
    """Check PyTorch installation."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            return True, f"PyTorch {torch.__version__} with CUDA {torch.version.cuda}"
        else:
            return True, f"PyTorch {torch.__version__} (CPU only)"
    except ImportError:
        return False, "PyTorch not found - install from https://pytorch.org"

def check_disk_space():
    """Check available disk space in current directory."""
    import shutil
    path = Path.cwd()
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    if free_gb >= 50:
        return True, f"{free_gb:.1f} GB free (recommended: 50+ GB)"
    else:
        return False, f"{free_gb:.1f} GB free (WARNING: need 50+ GB for full dataset)"

def check_config_file():
    """Check if config.yaml exists and is valid."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        return False, "config.yaml not found"
    
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        required = ["resize", "bins", "window_ms", "bucket_root", "raw_local_root"]
        missing = [k for k in required if k not in cfg]
        if missing:
            return False, f"Missing keys in config.yaml: {missing}"
        
        return True, "config.yaml valid"
    except Exception as e:
        return False, f"Error reading config.yaml: {e}"

def check_gcloud_auth():
    """Check if gcloud is authenticated (optional, for upload)."""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and "ACTIVE" in result.stdout:
            return True, "gcloud authenticated (ready for upload)"
        else:
            return False, "gcloud not authenticated - run: gcloud auth application-default login"
    except FileNotFoundError:
        return False, "gcloud CLI not installed (optional, only needed for GCP upload)"
    except Exception as e:
        return False, f"Could not check gcloud: {e}"

def main():
    print("=" * 60)
    print("DSEC Preprocessing Environment Check")
    print("=" * 60)
    print()
    
    checks = [
        ("Python version", check_python_version),
        ("Config file (config.yaml)", check_config_file),
        ("Disk space", check_disk_space),
        ("PyTorch", check_torch),
        ("numpy", lambda: check_module("numpy")),
        ("PIL (Pillow)", lambda: check_module("PIL")),
        ("yaml", lambda: check_module("yaml")),
        ("tonic", lambda: check_module("tonic")),
        ("h5py", lambda: check_module("h5py")),
        ("hdf5plugin", lambda: check_module("hdf5plugin")),
    ]
    
    results = []
    for description, check_func in checks:
        results.append(check_item(description, check_func))
        print()
    
    print("=" * 60)
    print("Optional (for GCP upload)")
    print("=" * 60)
    print()
    
    optional_checks = [
        ("google-cloud-storage", lambda: check_module("google.cloud.storage")),
        ("gcloud authentication", check_gcloud_auth),
    ]
    
    for description, check_func in optional_checks:
        check_item(description, check_func)
        print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"\033[92m✓ All checks passed ({passed}/{total})\033[0m")
        print("\nYou're ready to run:")
        print("  python preprocess_dsec.py --split train --max-samples 5")
    else:
        print(f"\033[91m✗ {total - passed} checks failed ({passed}/{total} passed)\033[0m")
        print("\nPlease fix the issues above before running preprocessing.")
        print("Install missing packages with:")
        print("  pip install -r requirements.txt")
    
    print()

if __name__ == "__main__":
    main()
