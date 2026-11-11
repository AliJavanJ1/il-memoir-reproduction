"""Small neural building blocks shared by the learners (f-PG policy, FB networks).

Learner-safe: nothing here reads a reward, a goal or a transition matrix.

Reproducibility (CLAUDE.md).  Every network is initialized from an explicit
CPU ``torch.Generator`` seeded by :func:`init_seed` (a value drawn from
``common.make_rng(seed, *labels)``), then moved to the requested device, so the
initial weights depend only on the seed and the stream labels, not on torch's
global random state or on the device.

Activations (U-12 and U-14, decided in Prompt 3B).  Only C^1 activations are
offered: ``tanh``, ``silu`` and ``elu``.  The gradient results FPG-1 to FPG-5
(theory Remark 3.3), FB-8(b)-(c), FB-9, FBIL-4 and FBIL-5 are proved only for
networks with C^1 activations; ReLU is deliberately not available.
``LayerNorm`` (with its positive ``eps``) is C^infinity.

Observations.  Finite mazes give integer state indices, which are one-hot
encoded; the continuous maze gives raw positions ``(x, y)`` in [0,1]^2, which are
fed to the networks unchanged (spec section 6, "raw (x, y)"; U-11).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from .common import make_rng
from .env_continuous import ContinuousMaze
from .env_tabular import GridMaze

ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU, "elu": nn.ELU}
DTYPES = {"float32": torch.float32, "float64": torch.float64}


def init_seed(seed: int, *labels: Union[str, int]) -> int:
    """A 62-bit integer seed for torch, drawn from the independent stream ``make_rng(seed, *labels)``."""
    return int(make_rng(seed, *labels).integers(0, 2**62))


def torch_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def resolve_device(device: Union[str, torch.device, None]) -> torch.device:
    """``cpu`` or ``cuda``; ``auto`` picks CUDA when available.  A CUDA request without CUDA raises."""
    if device is None or str(device) == "cpu":
        return torch.device("cpu")
    if str(device) == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return dev


def dtype_of(name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    if name not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {name!r}")
    return DTYPES[name]


def dtype_name(dtype: torch.dtype) -> str:
    for key, value in DTYPES.items():
        if value == dtype:
            return key
    raise ValueError(f"unsupported dtype {dtype}")


@torch.no_grad()
def init_linear(layer: nn.Linear, generator: torch.Generator, scale: float = 1.0) -> None:
    """``U(-k, k)`` with ``k = 1/sqrt(fan_in)`` for weight and bias (torch's default range), times ``scale``."""
    bound = 1.0 / math.sqrt(layer.in_features)
    w = torch.empty(layer.weight.shape, dtype=layer.weight.dtype).uniform_(-bound, bound, generator=generator)
    b = torch.empty(layer.bias.shape, dtype=layer.bias.dtype).uniform_(-bound, bound, generator=generator)
    layer.weight.copy_(scale * w)
    layer.bias.copy_(scale * b)


class MLP(nn.Module):
    """``in -> [hidden] * n -> out`` with C^1 activations.

    With ``first_layer_norm`` the first hidden layer is ``Linear -> LayerNorm -> tanh``
    (FI24 D.3, p. 28: "a layer normalization and Tanh activation in the first
    layer"); the later hidden layers use ``activation``.  ``out_scale`` multiplies
    the initial output layer (a small value makes an initial softmax policy
    close to uniform).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int],
        out_dim: int,
        activation: str,
        generator: torch.Generator,
        first_layer_norm: bool = False,
        out_scale: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)} (C^1 only), got {activation!r}")
        if not hidden:
            raise ValueError("at least one hidden layer is required")
        layers = []
        dims = [int(in_dim)] + [int(h) for h in hidden]
        for k in range(len(hidden)):
            lin = nn.Linear(dims[k], dims[k + 1], dtype=dtype)
            init_linear(lin, generator)
            layers.append(lin)
            if k == 0 and first_layer_norm:
                layers.append(nn.LayerNorm(dims[k + 1], dtype=dtype))
                layers.append(nn.Tanh())
            else:
                layers.append(ACTIVATIONS[activation]())
        out = nn.Linear(dims[-1], int(out_dim), dtype=dtype)
        init_linear(out, generator, scale=out_scale)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ObsSpec:
    """How observations are turned into network inputs: ``finite`` (one-hot) or ``continuous`` (raw x, y)."""

    kind: str
    n_states: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("finite", "continuous"):
            raise ValueError("ObsSpec.kind must be 'finite' or 'continuous'")
        if self.kind == "finite" and self.n_states < 1:
            raise ValueError("a finite ObsSpec needs n_states >= 1")

    @classmethod
    def from_env(cls, env: Any) -> "ObsSpec":
        if isinstance(env, GridMaze):
            return cls("finite", int(env.n_states))
        if isinstance(env, ContinuousMaze):
            return cls("continuous", 0)
        raise TypeError(f"unsupported environment type {type(env).__name__}")

    @property
    def finite(self) -> bool:
        return self.kind == "finite"

    @property
    def dim(self) -> int:
        return self.n_states if self.finite else 2

    def as_array(self, obs: Any) -> np.ndarray:
        """Validated numpy observations: int64 indices ``(...)`` or float64 positions ``(..., 2)``."""
        if self.finite:
            arr = np.asarray(obs)
            if arr.dtype.kind not in "iu":
                raise ValueError("finite observations must be integer state indices")
            arr = arr.astype(np.int64)
            if arr.size and (arr.min() < 0 or arr.max() >= self.n_states):
                raise ValueError("state index out of range")
            return arr
        arr = np.asarray(obs, dtype=np.float64)
        if arr.shape[-1:] != (2,) or not np.all(np.isfinite(arr)):
            raise ValueError("continuous observations must be finite positions of shape (..., 2)")
        return arr

    def encode(self, obs: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Network input: one-hot rows (finite) or the raw positions (continuous)."""
        if isinstance(obs, torch.Tensor):
            if self.finite:
                return nn.functional.one_hot(obs.to(device=device, dtype=torch.int64), self.n_states).to(dtype)
            return obs.to(device=device, dtype=dtype)
        arr = self.as_array(obs)
        if self.finite:
            idx = torch.as_tensor(arr, device=device)
            return nn.functional.one_hot(idx, self.n_states).to(dtype)
        return torch.as_tensor(arr, device=device).to(dtype)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "n_states": int(self.n_states)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObsSpec":
        return cls(str(d["kind"]), int(d.get("n_states", 0)))


def params_fingerprint(module: nn.Module) -> torch.Tensor:
    """A detached float64 CPU copy of all parameters, concatenated (for freeze checks)."""
    with torch.no_grad():
        parts = [p.detach().to("cpu", torch.float64).reshape(-1) for p in module.parameters()]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)


def flat_grad(loss: torch.Tensor, params: Sequence[torch.Tensor], allow_unused: bool = False) -> np.ndarray:
    """Gradient of a scalar with respect to ``params`` as one float64 numpy vector."""
    grads = torch.autograd.grad(loss, list(params), allow_unused=allow_unused)
    out = []
    for g, p in zip(grads, params):
        out.append(np.zeros(p.numel()) if g is None else g.detach().to("cpu", torch.float64).reshape(-1).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move an optimizer's state tensors (Adam moments) to ``device`` after ``load_state_dict``."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device)


def rng_state(rng: np.random.Generator) -> Dict[str, Any]:
    """JSON/torch-safe copy of a PCG64 generator state (for checkpoints)."""
    st = rng.bit_generator.state
    return {
        "bit_generator": str(st["bit_generator"]),
        "state": {"state": str(st["state"]["state"]), "inc": str(st["state"]["inc"])},
        "has_uint32": int(st["has_uint32"]),
        "uinteger": int(st["uinteger"]),
    }


def set_rng_state(rng: np.random.Generator, saved: Dict[str, Any]) -> None:
    rng.bit_generator.state = {
        "bit_generator": saved["bit_generator"],
        "state": {"state": int(saved["state"]["state"]), "inc": int(saved["state"]["inc"])},
        "has_uint32": int(saved["has_uint32"]),
        "uinteger": int(saved["uinteger"]),
    }


def check_finite_tensor(x: torch.Tensor, what: str) -> None:
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(f"{what} is not finite")


def to_numpy64(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu", torch.float64).numpy()


def maybe_float(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(x)
