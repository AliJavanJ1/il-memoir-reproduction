"""Prepare (or verify) the frozen data of a suite: demonstrations, task-blind replay, held-out references.

Usage, from the project root:

    python scripts/prepare_data.py --config configs/core.yaml
    python scripts/prepare_data.py --config configs/core.yaml --check     # verify stored files only

Every declared seed, layout and goal of the suite is prepared in
``data/<data.id>/`` with ``manifest.json``.  Each item is regenerated
deterministically and compared with the stored file: identical content is
"verified", a missing item is written, and different content or a different
specification stops the script without overwriting anything (a changed data
definition needs a new ``data.id``).  Replay coverage of every state-action
pair (finite) or free cell and action (continuous) is checked.  Exit status 0
iff every item is verified or written.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # launched as `python scripts/...` from the project root
    sys.path.insert(0, str(ROOT))

from il_memoir import datasets as ds  # noqa: E402
from il_memoir.experiment import ConfigError, data_items, load_suite_config  # noqa: E402
from il_memoir.prepare import DataConflict, prepare_items  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="suite configuration, e.g. configs/core.yaml")
    parser.add_argument("--data-root", default=str(ROOT / "data"), help="data directory (default: data/)")
    parser.add_argument("--check", action="store_true", help="only verify the stored files against the manifest")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    try:
        suite = load_suite_config(args.config)
    except (ConfigError, ValueError, OSError) as exc:
        print(f"configuration error: {exc}")
        return 2
    root = Path(args.data_root)
    items = data_items(suite)
    print(f"suite {suite.name} ({suite.kind}), data id {suite.data_id}: {len(items)} item(s) in {root / suite.data_id}")
    try:
        if args.check:
            for key, item in items.items():
                ds.verify_item(root, suite.data_id, key, item.spec_sha256)
                print(f"  ok       {key}")
        else:
            prepare_items(suite, root)
    except (DataConflict, ds.DataError, KeyError) as exc:
        print(f"DATA ERROR: {exc}")
        return 1
    print(f"done in {time.perf_counter() - t0:.1f} s; manifest: {ds.manifest_path(root, suite.data_id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
