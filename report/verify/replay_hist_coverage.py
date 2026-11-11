#!/usr/bin/env python3
"""Coverage of the continuous replay buffers at two resolutions (closeout, October 2026; AUDIT.md M5).

Reads the stored final continuous replay buffers (data/final-v1/replay/continuous/<layout>/s<seed>.npz), checks the
content hash of each file against data/final-v1/manifest.json, and counts the visits of every pair (cell, action),
where the cell is that of the transition's start state:

* on the maze lattice (cells of side 1/7; the free lattice cells), the resolution of the coverage check recorded at
  data preparation and in the FB results (``min_cell_action_count``, il_memoir/fb.py ``replay_coverage``), and
* on the free-space histogram of the evaluation (20 x 20 bins of side 0.05, the bins of positive free area), with
  the evaluation's own histogram code (il_memoir.density.FreeSpaceHistogram, as in il_memoir/evaluation.py).

Writes report/verify/replay_hist_coverage.json, which report/verify/claims.py reads; prints a summary.  Uses NumPy,
SciPy and the project package (read only, no torch).  Run from the project root:

    python report/verify/replay_hist_coverage.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from il_memoir.config import load_environment_config  # noqa: E402
from il_memoir.density import FreeSpace, FreeSpaceHistogram  # noqa: E402
from il_memoir.env_tabular import make_layout  # noqa: E402

DATA = ROOT / "data" / "final-v1"
OUT = ROOT / "report" / "verify" / "replay_hist_coverage.json"
LOW = 10          # a (cell, action) pair with fewer visits than this is reported as thinly covered
FIELDS = ("obs", "actions", "next_obs", "segment", "step_in_segment")


def load_replay(path: Path, expected_sha256: str):
    # The replay file format of il_memoir/fb.py (ReplayBuffer.save/.sha256), read without importing torch: the content
    # hash covers dtype, shape and bytes of the five arrays in this order; it must equal the stored and the manifest hash.
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {k: np.array(data[k]) for k in FIELDS}
    h = hashlib.sha256()
    for k in FIELDS:
        a = np.ascontiguousarray(arrays[k])
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    if not (h.hexdigest() == meta.get("sha256") == expected_sha256):
        raise SystemExit(f"{path}: content hash {h.hexdigest()} differs from the stored or the manifest hash")
    return arrays


def main() -> int:
    cfg = load_environment_config(ROOT / "configs" / "environment.yaml")
    n_actions = len(cfg.action_names)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    out = {"source": "data/final-v1/manifest.json and the stored replay files; content hashes checked",
           "low_threshold": LOW, "action_names": list(cfg.action_names), "buffers": []}
    for key, e in sorted(manifest["items"].items()):
        spec = e["spec"]
        if e["kind"] != "replay" or spec["env"] != "continuous":
            continue
        arrays = load_replay(DATA / e["path"], e["content_sha256"])        # stops if the content hash differs
        obs = np.asarray(arrays["obs"], dtype=np.float64).reshape(-1, 2)
        acts = np.asarray(arrays["actions"], dtype=np.int64).reshape(-1)
        layout = make_layout(cfg, spec["layout"])
        # lattice resolution (as fb.replay_coverage: floor(grid_size * x), visited cells)
        g = int(cfg.grid_size)
        cells = np.clip(np.floor(obs * g).astype(np.int64), 0, g - 1)
        lat = np.zeros((g, g, n_actions), dtype=np.int64)
        np.add.at(lat, (cells[:, 0], cells[:, 1], acts), 1)
        visited = lat.sum(axis=2) > 0
        # histogram resolution (as the evaluation)
        hist = FreeSpaceHistogram(FreeSpace.from_layout(layout), int(cfg.histogram_bins))
        ids = hist.cell_ids(obs)
        cnt = np.zeros((hist.n_cells, n_actions), dtype=np.int64)
        np.add.at(cnt, (ids, acts), 1)
        bin_area = (1.0 / int(cfg.histogram_bins)) ** 2
        low = np.argwhere(cnt < LOW)
        frac = sorted(float(hist.free_area[c] / bin_area) for c, _ in low)
        zero = [{"bin": [int(v) for v in hist.cells[c]], "action": int(a), "action_name": cfg.action_names[int(a)],
                 "free_area_fraction": float(hist.free_area[c] / bin_area)} for c, a in np.argwhere(cnt == 0)]
        out["buffers"].append({
            "key": key, "seed": int(spec["seed"]), "layout": spec["layout"], "path": e["path"],
            "content_sha256": e["content_sha256"], "content_hash_checked": True, "n_transitions": int(acts.size),
            "lattice": {"cell_side": 1.0 / g, "cells_visited": int(visited.sum()),
                        "min_cell_action_count": int(lat[visited].min()),
                        "manifest_min_cell_action_count": e.get("coverage", {}).get("min_cell_action_count")},
            "histogram": {"bins_per_axis": int(cfg.histogram_bins), "n_cells": int(hist.n_cells),
                          "n_pairs": int(cnt.size), "min_count": int(cnt.min()), "median_count": float(np.median(cnt)),
                          "n_pairs_below_low": int(low.shape[0]),
                          "free_area_fraction_of_low_bins": {"min": min(frac) if frac else None,
                                                             "max": max(frac) if frac else None},
                          "zero_count_pairs": zero}})
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for b in out["buffers"]:
        h, la = b["histogram"], b["lattice"]
        print(f"{b['key']}: lattice min {la['min_cell_action_count']} (manifest {la['manifest_min_cell_action_count']}, "
              f"{la['cells_visited']} cells) | histogram {h['n_cells']} cells: min {h['min_count']}, median "
              f"{h['median_count']:g}, {h['n_pairs_below_low']} pairs below {LOW} (free-area fraction "
              f"{h['free_area_fraction_of_low_bins']['min']:.3f}-{h['free_area_fraction_of_low_bins']['max']:.3f}); "
              f"zero-count pairs {[(z['bin'], z['action_name']) for z in h['zero_count_pairs']]}")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
