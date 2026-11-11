"""Device policy, hardware records, synchronized timers and memory peaks (bookkeeping only).

Device policy (INTERFACE.md)
----------------------------
``resolve_run_device`` turns ``--device auto|cpu|cuda`` into the device a job
runs on, and says why:

* ``cpu`` is always honoured;
* ``cuda`` uses CUDA if it is available; otherwise it falls back to the CPU
  with a visible message (recorded as a fallback);
* ``auto`` uses, in this order, the device frozen in the suite manifest (a
  resumed suite never changes device silently: if that device is not
  available now, the call fails and asks for an explicit ``--device``), then
  the device chosen by the pilot (``runs/benchmark.json``, only if it was
  recorded on this computer), then -- outside a suite -- CUDA if available,
  else the CPU.  ``run_suite.py`` refuses ``auto`` before the pilot exists;
* in a Slurm allocation (``SLURM_JOB_ID`` set) the visible GPUs
  (``CUDA_VISIBLE_DEVICES``) are honoured by torch; on a login node (Slurm
  commands present, no allocation) GPU availability is never inferred:
  ``auto`` means CPU and an explicit ``cuda`` is refused.

Nothing here writes ``runs/environment_laptop.json`` (the hardware record of
your laptop, written only by ``scripts/diagnose.py``).
"""
from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

from .logging import host_fingerprint, read_json, utc_now

LAPTOP_RECORD = "environment_laptop.json"     # never written by the package
PILOT_RECORD = "benchmark.json"


# =============================================================================
# scheduler context
# =============================================================================
def scheduler_context() -> Dict[str, Any]:
    """Slurm allocation or login node?  Only environment variables and the PATH are read."""
    env = os.environ
    in_allocation = bool(env.get("SLURM_JOB_ID"))
    slurm_present = bool(shutil.which("sbatch") or env.get("SLURM_CONF") or env.get("SLURM_CLUSTER_NAME"))
    return {
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "slurm_job_gpus": env.get("SLURM_JOB_GPUS") or env.get("SLURM_STEP_GPUS"),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "in_allocation": in_allocation,
        "login_node": slurm_present and not in_allocation,
    }


def cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


# =============================================================================
# pilot choice
# =============================================================================
def read_pilot_choice(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """The pilot's device choice if ``path`` exists, is complete and was recorded on this computer."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        rec = read_json(p)
    except (OSError, ValueError):
        return None
    choice = rec.get("device_choice") or {}
    if not rec.get("complete") or choice.get("device") not in ("cpu", "cuda"):
        return None
    host = rec.get("host") or {}
    here = host_fingerprint()
    return {
        "device": choice["device"],
        "path": str(p),
        "recorded_utc": rec.get("recorded_utc"),
        "same_host": host.get("hostname") == here["hostname"] and host.get("system") == here["system"],
        "host": host,
    }


@dataclass
class DeviceDecision:
    device: str                     # "cpu" or "cuda"
    requested: str
    source: str                     # explicit | suite-frozen | pilot | default
    fallback: bool = False
    message: str = ""
    scheduler: Dict[str, Any] = field(default_factory=dict)
    recorded_utc: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeviceError(RuntimeError):
    pass


def resolve_run_device(
    requested: str,
    frozen: Optional[str] = None,
    pilot: Optional[Dict[str, Any]] = None,
    require_pilot_for_auto: bool = False,
    cuda_ok: Optional[bool] = None,
    sched: Optional[Dict[str, Any]] = None,
) -> DeviceDecision:
    """Apply the device policy of the module docstring.  ``cuda_ok`` and ``sched`` can be injected by tests."""
    if requested not in ("auto", "cpu", "cuda"):
        raise DeviceError(f"--device must be auto, cpu or cuda, got {requested!r}")
    sched = scheduler_context() if sched is None else sched
    login = bool(sched.get("login_node"))
    have_cuda = (cuda_available() if cuda_ok is None else bool(cuda_ok)) and not login
    if requested == "cpu":
        return DeviceDecision("cpu", requested, "explicit", scheduler=sched, message="--device cpu (explicit)")
    if requested == "cuda":
        if login:
            raise DeviceError("this looks like a cluster login node (Slurm commands present, no SLURM_JOB_ID): "
                              "GPU availability is not inferred here; run inside an allocation (sbatch/srun)")
        if have_cuda:
            return DeviceDecision("cuda", requested, "explicit", scheduler=sched, message="--device cuda (explicit)")
        return DeviceDecision("cpu", requested, "explicit", fallback=True, scheduler=sched,
                              message="WARNING: --device cuda was requested but CUDA is not available; "
                                      "falling back to the CPU (recorded in the run metadata)")
    # auto
    if frozen is not None:
        if frozen == "cuda" and not have_cuda:
            raise DeviceError("this suite was started on cuda (frozen in its manifest) but CUDA is not available now; "
                              "pass --device cpu explicitly to continue on the CPU (recorded per run; float32 numerics "
                              "differ between devices, METHOD_MAP K2)")
        return DeviceDecision(frozen, requested, "suite-frozen", scheduler=sched,
                              message=f"--device auto: using the device frozen in the suite manifest ({frozen})")
    if pilot is not None and pilot.get("same_host"):
        dev = pilot["device"]
        if dev == "cuda" and not have_cuda:
            return DeviceDecision("cpu", requested, "pilot", fallback=True, scheduler=sched,
                                  message="WARNING: the pilot chose cuda but CUDA is not available; falling back to the CPU")
        return DeviceDecision(dev, requested, "pilot", scheduler=sched,
                              message=f"--device auto: using the pilot's measured choice ({dev}, {pilot.get('path')})")
    if require_pilot_for_auto:
        extra = ""
        if pilot is not None and not pilot.get("same_host"):
            extra = f" ({pilot.get('path')} was recorded on another computer: {pilot.get('host')})"
        raise DeviceError("--device auto needs the pilot's saved device choice (runs/benchmark.json from "
                          "scripts/benchmark.py on this computer)" + extra + "; run the pilot first or pass --device cpu/cuda")
    if login:
        return DeviceDecision("cpu", requested, "default", scheduler=sched,
                              message="--device auto on a login node: CPU (GPU availability is not inferred here)")
    if have_cuda:
        return DeviceDecision("cuda", requested, "default", scheduler=sched,
                              message="--device auto: no pilot choice yet; CUDA is available and is used")
    return DeviceDecision("cpu", requested, "default", scheduler=sched,
                          message="--device auto: no pilot choice yet and CUDA is not available; using the CPU")


def set_cpu_threads(n: int) -> int:
    """At most ``n`` intra-op CPU threads for torch (spec section 7: at most 4 until measured)."""
    import torch

    n = max(1, int(n))
    torch.set_num_threads(n)
    return torch.get_num_threads()


# =============================================================================
# hardware record
# =============================================================================
def hardware_record(device: str = "cpu") -> Dict[str, Any]:
    """Interpreter, CPU, RAM, torch, CUDA and GPU of this process; host labels are recorded, not inferred."""
    import torch

    rec: Dict[str, Any] = {
        "recorded_utc": utc_now(),
        "host": host_fingerprint(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "scheduler": scheduler_context(),
        "in_container": os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        rec["ram_total_mib"] = round(vm.total / 2**20, 1)
        rec["ram_available_mib"] = round(vm.available / 2**20, 1)
    except ImportError:                                   # pragma: no cover
        rec["ram_total_mib"] = None
        rec["ram_available_mib"] = None
    if torch.cuda.is_available():
        rec["gpus"] = []
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            rec["gpus"].append({"index": i, "name": prop.name, "vram_mib": round(prop.total_memory / 2**20, 1)})
    return rec


# =============================================================================
# timers and memory
# =============================================================================
def sync(device: Union[str, Any]) -> None:
    """``torch.cuda.synchronize()`` on a CUDA device, so that wall-clock readings include queued kernels."""
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


class PhaseTimer:
    """Accumulated, synchronized wall time per named phase (seconds)."""

    def __init__(self, device: str = "cpu", initial: Optional[Dict[str, float]] = None):
        self.device = device
        self.totals: Dict[str, float] = dict(initial or {})

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            sync(self.device)
            self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + float(seconds)

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in sorted(self.totals.items())}


def rss_mib() -> Optional[float]:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except ImportError:                                   # pragma: no cover
        return None


def peak_rss_mib() -> Optional[float]:
    """Peak resident memory of this process so far (MiB): ``peak_wset`` on Windows, ``ru_maxrss`` elsewhere."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        if hasattr(info, "peak_wset"):
            return round(info.peak_wset / 2**20, 1)
    except ImportError:                                   # pragma: no cover
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1.0 if sys.platform == "darwin" else 1024.0     # bytes on macOS, KiB on Linux
        return round(peak * scale / 2**20, 1)
    except (ImportError, OSError):                        # pragma: no cover
        return None


def reset_cuda_peak(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()


def cuda_peak_mib(device: str) -> Optional[float]:
    if str(device).startswith("cuda"):
        import torch

        return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return None


def memory_record(device: str) -> Dict[str, Optional[float]]:
    return {"rss_mib": rss_mib(), "process_peak_rss_mib": peak_rss_mib(), "cuda_peak_allocated_mib": cuda_peak_mib(device)}


def hostname() -> str:
    return socket.gethostname()
