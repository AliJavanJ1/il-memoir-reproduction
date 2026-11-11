"""Record the actual interpreter, RAM, PyTorch and CUDA without training.

Run this in native Windows PowerShell, not only in a Claude sandbox.
Only standard-library dependencies are required. psutil and torch are optional.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def diagnose() -> dict:
    report = {
        "schema_version": 1,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "logical_cpus": os.cpu_count(),
        "disk_free_gib": round(shutil.disk_usage(Path.cwd()).free / 2**30, 2),
        "packages": {},
        "torch_imported": False,
        "cuda_available": False,
        "cuda_test_passed": False,
        "devices": [],
        "errors": [],
    }
    for package in ("torch", "numpy", "scipy", "matplotlib", "PyYAML", "pytest", "psutil"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    try:
        import psutil
        ram = psutil.virtual_memory()
        report["ram_total_gib"] = round(ram.total / 2**30, 2)
        report["ram_available_gib"] = round(ram.available / 2**30, 2)
    except ImportError:
        report["ram_total_gib"] = None
        report["ram_available_gib"] = None

    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            report["nvidia_smi"] = {
                "exit_code": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            report["errors"].append("nvidia-smi: " + str(error))
    else:
        report["nvidia_smi"] = None

    try:
        import torch
        report["torch_imported"] = True
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                report["devices"].append({
                    "index": index,
                    "name": properties.name,
                    "vram_gib": round(properties.total_memory / 2**30, 2),
                })
            with torch.no_grad():
                matrix = torch.eye(64, device="cuda:0")
                product = matrix @ matrix
                torch.cuda.synchronize()
                report["cuda_test_passed"] = bool(torch.allclose(product, matrix))
                del product, matrix
    except ImportError as error:
        report["errors"].append("PyTorch unavailable: " + str(error))
    except Exception as error:
        report["errors"].append("PyTorch/CUDA diagnostic: " + repr(error))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/environment_laptop.json"))
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = diagnose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.output}")
    if args.require_cuda and not report["cuda_test_passed"]:
        print("CUDA verification failed; the report was saved for troubleshooting.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
