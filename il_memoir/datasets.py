"""Frozen data on disk: layout, manifest, content hashes and loaders (INTERFACE ``data/``).

``data/<data_id>/`` holds the demonstrations, the task-blind replay buffers and
the held-out reference rollouts of every suite that declares ``data.id =
<data_id>``, with ``manifest.json``: for each item its specification (what
generated it, with which seed and stream), the SHA-256 of its specification,
the SHA-256 of its *content* (array bytes; independent of the zip timestamps
inside ``.npz`` files) and the SHA-256 of the file.

Generation lives in :mod:`il_memoir.prepare`.  This module only reads, so a
learner-side runner can import it without importing the expert, the oracle
or the evaluation reward:

* :func:`load_demo_states` returns the demonstration **states** only (the
  learner view; METHOD_MAP section 1).  Expert actions are read only by
  :func:`load_demonstrations`, for evaluation diagnostics;
* :func:`replay_path` locates a replay buffer; ``fb.ReplayBuffer.load`` reads
  it and checks its own content hash;
* :func:`load_reference_states` returns held-out expert rollouts for
  evaluation only (never a learner input).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from .logging import PROJECT_ROOT, read_json, replace_file, sha256_array, sha256_file, sha256_text, utc_now, write_json_atomic

DATA_SCHEMA = "il_memoir/data-v1"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
ARRAY_FORMAT = "il_memoir/arrays-v1"


class DataError(RuntimeError):
    pass


def data_dir(root: Union[str, Path], data_id: str) -> Path:
    return Path(root) / data_id


def manifest_path(root: Union[str, Path], data_id: str) -> Path:
    return data_dir(root, data_id) / "manifest.json"


def item_path(root: Union[str, Path], data_id: str, key: str) -> Path:
    return data_dir(root, data_id) / (key + ".npz")


def load_manifest(root: Union[str, Path], data_id: str) -> Dict[str, Any]:
    path = manifest_path(root, data_id)
    if not path.exists():
        return {"schema": DATA_SCHEMA, "data_id": data_id, "created_utc": utc_now(), "items": {}}
    rec = read_json(path)
    if rec.get("schema") != DATA_SCHEMA or rec.get("data_id") != data_id:
        raise DataError(f"{path} is not a data manifest for {data_id!r}")
    return rec


def save_manifest(root: Union[str, Path], data_id: str, manifest: Mapping[str, Any]) -> None:
    rec = dict(manifest)
    rec["updated_utc"] = utc_now()
    write_json_atomic(manifest_path(root, data_id), rec)


def arrays_sha256(arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> str:
    """Content hash: array names, dtypes, shapes and bytes, plus the canonical metadata."""
    parts = [f"{name}:{sha256_array(np.asarray(arrays[name]))}" for name in sorted(arrays)]
    parts.append("meta:" + json.dumps(meta, sort_keys=True, separators=(",", ":")))
    return sha256_text("\n".join(parts))


def save_arrays(path: Union[str, Path], arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> str:
    """Compressed ``.npz`` with a JSON metadata entry; returns the content hash."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = arrays_sha256(arrays, meta)
    full_meta = dict(meta, format=ARRAY_FORMAT, content_sha256=content)
    tmp = dest.with_name(dest.name + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(full_meta, sort_keys=True)), **{k: np.asarray(v) for k, v in arrays.items()})
    replace_file(tmp, dest)
    return content


def load_arrays(path: Union[str, Path], expected_content_sha256: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Arrays and metadata of a file written by :func:`save_arrays`; the content hash is recomputed and checked."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {k: np.array(data[k]) for k in data.files if k != "meta"}
    stored = meta.pop("content_sha256", None)
    if meta.pop("format", None) != ARRAY_FORMAT:
        raise DataError(f"{path}: not an il_memoir array file")
    actual = arrays_sha256(arrays, meta)
    if stored != actual:
        raise DataError(f"{path}: content hash mismatch (stored {stored}, recomputed {actual})")
    if expected_content_sha256 is not None and actual != expected_content_sha256:
        raise DataError(f"{path}: content hash {actual} differs from the expected {expected_content_sha256}")
    return arrays, meta


# =============================================================================
# manifest lookups
# =============================================================================
def manifest_entry(root: Union[str, Path], data_id: str, key: str) -> Dict[str, Any]:
    man = load_manifest(root, data_id)
    entry = man["items"].get(key)
    if entry is None:
        raise DataError(f"data item {key!r} is missing from {manifest_path(root, data_id)}; run scripts/prepare_data.py")
    return entry


def verify_item(root: Union[str, Path], data_id: str, key: str, spec_sha256: Optional[str] = None) -> Dict[str, Any]:
    """The manifest entry of an item after checking that its file exists, its file hash and its specification."""
    entry = manifest_entry(root, data_id, key)
    path = item_path(root, data_id, key)
    if not path.exists():
        raise DataError(f"data file {path} is missing")
    if spec_sha256 is not None and entry.get("spec_sha256") != spec_sha256:
        raise DataError(f"data item {key!r} was generated from another specification; use a new data.id")
    actual = sha256_file(path)
    if actual != entry.get("file_sha256"):
        raise DataError(f"data file {path} changed after preparation (file hash {actual} != {entry.get('file_sha256')})")
    return entry


# =============================================================================
# loaders
# =============================================================================
def load_demo_states(root: Union[str, Path], data_id: str, key: str, content_sha256: Optional[str] = None) -> np.ndarray:
    """Demonstration states ``s_0..s_H`` (shape ``(N_E, H+1)`` or ``(N_E, H+1, 2)``): the learner view, no actions."""
    arrays, meta = load_arrays(item_path(root, data_id, key), content_sha256)
    if meta.get("kind") != "demos":
        raise DataError(f"{key} is not a demonstration item")
    return np.array(arrays["states"], copy=True)


def load_demonstrations(root: Union[str, Path], data_id: str, key: str,
                        content_sha256: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """States, expert actions and metadata: **evaluation and diagnostics only** (never a learner input)."""
    arrays, meta = load_arrays(item_path(root, data_id, key), content_sha256)
    if meta.get("kind") != "demos":
        raise DataError(f"{key} is not a demonstration item")
    return arrays["states"], arrays["actions"], meta


def load_reference_states(root: Union[str, Path], data_id: str, key: str,
                          content_sha256: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Held-out expert reference rollouts (states): **evaluation only**."""
    arrays, meta = load_arrays(item_path(root, data_id, key), content_sha256)
    if meta.get("kind") != "reference":
        raise DataError(f"{key} is not a reference item")
    return arrays["states"], meta


def replay_path(root: Union[str, Path], data_id: str, key: str) -> Path:
    """Path of a replay buffer (``fb.ReplayBuffer.load`` verifies its content hash)."""
    return item_path(root, data_id, key)
