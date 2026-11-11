"""Reload and resume checks on saved runs (used by ``scripts/smoke.py`` and the tests; bookkeeping only).

* :func:`reload_checks` reloads every saved policy, FB snapshot and latent of a
  suite and compares the action probabilities with those the job recorded
  in-process (``outputs/probe.npz``): on the CPU they must be identical; on CUDA
  they must agree to float32 precision (METHOD_MAP K2).
* :func:`compare_runs` compares two run directories of the same suite (one run
  uninterrupted, one interrupted and resumed): the final parameters, latents and
  evaluation metrics of every job must be identical on the CPU.
* :func:`output_hashes_intact` checks that evaluation left every training output
  untouched (SHA-256 recorded at training time).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch

from .fb import FBModel
from .fpg import policy_callable, policy_from_record
from .logging import JsonlLog, load_torch, read_json, sha256_file


def _final_policy(job_dir: Path, res: Dict[str, Any]) -> Path:
    last = max(res["outputs"]["policies"], key=lambda o: o["iteration"])
    return job_dir / last["path"]


def reload_checks(runs_root: Union[str, Path], manifest: Dict[str, Any], atol: float = 0.0) -> Dict[str, bool]:
    """``{check name: passed}`` for every training job of ``manifest`` (``atol = 0`` means bitwise)."""
    runs_root = Path(runs_root)
    out: Dict[str, bool] = {}
    for j in manifest["jobs"]:
        d = runs_root / "jobs" / j["job_id"]
        if not (d / "result.json").exists():
            continue
        res = read_json(d / "result.json")
        probe_path = d / "outputs" / "probe.npz"
        if j["type"] not in ("fpg", "fb_pretrain", "adapt") or not probe_path.exists():
            continue
        with np.load(probe_path) as data:
            probe = {k: np.array(data[k]) for k in data.files}
        states = probe["states"]
        if j["type"] == "fpg":
            pol = policy_from_record(load_torch(_final_policy(d, res)), "cpu")
            got = policy_callable(pol)(states)
            out[f"reload_fpg_policy:{j['job_id']}"] = bool(np.allclose(got, probe["probs"], rtol=0, atol=atol))
        elif j["type"] == "fb_pretrain":
            snap = max(res["outputs"]["snapshots"], key=lambda s: s["update"])
            model = FBModel.load(d / snap["path"], "cpu")
            got = model.numpy_policy(probe["z"])(states)
            out[f"reload_fb_model:{j['job_id']}"] = bool(np.allclose(got, probe["probs"], rtol=0, atol=atol))
        else:
            fb_dir = runs_root / "jobs" / res["outputs"]["fb_job"]
            lat = read_json(d / res["outputs"]["latents"]["path"])
            rec = max(lat["records"], key=lambda r: r["fb_update"])
            model = FBModel.load(fb_dir / rec["fb_model"]["path"], "cpu")
            ok = True
            for kind in ("er", "dm"):
                z = np.asarray(rec[kind]["z"])
                ok &= bool(np.array_equal(z, probe[f"{kind}_z"]))
                ok &= bool(np.allclose(model.numpy_policy(z)(states), probe[f"{kind}_probs"], rtol=0, atol=atol))
            out[f"reload_latents:{j['job_id']}"] = ok
    return out


def output_hashes_intact(runs_root: Union[str, Path], manifest: Dict[str, Any]) -> Dict[str, bool]:
    runs_root = Path(runs_root)
    out = {}
    for j in manifest["jobs"]:
        d = runs_root / "jobs" / j["job_id"]
        if j["type"] not in ("fpg", "fb_pretrain", "adapt") or not (d / "result.json").exists():
            continue
        res = read_json(d / "result.json")
        files: List[Dict[str, str]] = []

        def walk(o):
            if isinstance(o, dict):
                if isinstance(o.get("path"), str) and isinstance(o.get("sha256"), str):
                    files.append(o)
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(res["outputs"])
        out[f"outputs_unchanged_by_evaluation:{j['job_id']}"] = all(
            (d / f["path"]).exists() and sha256_file(d / f["path"]) == f["sha256"] for f in files)
    return out


def _state_tensors(path: Path) -> Dict[str, torch.Tensor]:
    rec = load_torch(path)
    if "state_dict" in rec:
        return dict(rec["state_dict"])
    return {**{f"F.{k}": v for k, v in rec["F"].items()}, **{f"B.{k}": v for k, v in rec["B"].items()},
            "gram": rec["gram"]}


def _max_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    if set(a) != set(b):
        return float("inf")
    return max(float((a[k].double() - b[k].double()).abs().max()) if a[k].numel() else 0.0 for k in a)


def compare_runs(runs_a: Union[str, Path], runs_b: Union[str, Path], manifest: Dict[str, Any], atol: float = 0.0) -> Dict[str, Any]:
    """Final outputs of every job in two runs of the same suite; ``atol = 0`` requires bitwise equality."""
    a_root, b_root = Path(runs_a), Path(runs_b)
    checks: Dict[str, bool] = {}
    details: Dict[str, float] = {}
    for j in manifest["jobs"]:
        da, db = a_root / "jobs" / j["job_id"], b_root / "jobs" / j["job_id"]
        ra, rb = read_json(da / "result.json"), read_json(db / "result.json")
        if j["type"] == "fpg":
            diff = _max_diff(_state_tensors(_final_policy(da, ra)), _state_tensors(_final_policy(db, rb)))
            la = [{k: v for k, v in r.items() if not k.startswith("time_")} for r in JsonlLog(da / "log.jsonl").records()]
            lb = [{k: v for k, v in r.items() if not k.startswith("time_")} for r in JsonlLog(db / "log.jsonl").records()]
            checks[f"resume_fpg_log_equal:{j['job_id']}"] = la == lb if atol == 0 else len(la) == len(lb)
        elif j["type"] == "fb_pretrain":
            sa = max(ra["outputs"]["snapshots"], key=lambda s: s["update"])
            sb = max(rb["outputs"]["snapshots"], key=lambda s: s["update"])
            diff = _max_diff(_state_tensors(da / sa["path"]), _state_tensors(db / sb["path"]))
            la = [{k: v for k, v in r.items() if k != "wall_s"} for r in JsonlLog(da / "log.jsonl").records()]
            lb = [{k: v for k, v in r.items() if k != "wall_s"} for r in JsonlLog(db / "log.jsonl").records()]
            checks[f"resume_fb_log_equal:{j['job_id']}"] = la == lb if atol == 0 else len(la) == len(lb)
        elif j["type"] == "adapt":
            la = read_json(da / ra["outputs"]["latents"]["path"])["records"]
            lb = read_json(db / rb["outputs"]["latents"]["path"])["records"]
            diff = max(float(np.max(np.abs(np.asarray(x[k]["z"]) - np.asarray(y[k]["z"])))) for x, y in zip(la, lb) for k in ("er", "dm"))
        else:
            ea = JsonlLog(da / ra["outputs"]["evaluations"]["path"]).records()
            eb = JsonlLog(db / rb["outputs"]["evaluations"]["path"]).records()
            diff = 0.0
            for x, y in zip(ea, eb):
                for k, v in x["metrics"].items():
                    if isinstance(v, float) and isinstance(y["metrics"].get(k), float):
                        diff = max(diff, abs(v - y["metrics"][k]))
            if len(ea) != len(eb):
                diff = float("inf")
        details[j["job_id"]] = diff
        checks[f"resume_equals_uninterrupted:{j['job_id']}"] = diff <= atol
    return {"checks": checks, "max_abs_diff": details}
