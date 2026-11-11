"""Suite configurations, the frozen seed manifest and the job graph (Prompt 3C).

The plans must match PROJECT_SPEC section 7 exactly (counts of runs per method,
environment and seed), the compact ablations and the reduced core must reuse
the core's jobs instead of retraining them, identifiers must be deterministic
hashes of everything that can change a job's outputs, and the random streams
must be independent where the spec requires it and shared (common random
numbers) where variants are compared.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from il_memoir import experiment as ex
from il_memoir.logging import sha256_json

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["core", "compact_ablations", "reduced", "pilot", "smoke"]


@pytest.fixture(scope="module")
def suites():
    return {n: ex.load_suite_config(ROOT / "configs" / f"{n}.yaml") for n in CONFIGS}


def _jobs(suite, jtype=None, **labels):
    out = []
    for j in ex.expand_jobs(suite):
        if jtype is not None and j.type != jtype:
            continue
        if all(j.labels.get(k) == v for k, v in labels.items()):
            out.append(j)
    return out


def test_core_plan_matches_the_spec(suites):
    core = suites["core"]
    assert core.kind == "final" and core.seeds == [0, 1, 2]
    assert len(_jobs(core, "fpg", env="finite")) == 18 and len(_jobs(core, "fpg", env="continuous")) == 9
    assert len(_jobs(core, "fb_pretrain", env="finite")) == 6 and len(_jobs(core, "fb_pretrain", env="continuous")) == 3
    assert len(_jobs(core, "adapt")) == 27 and len(_jobs(core, "eval")) == 54 and len(_jobs(core, "eval_expert")) == 27
    # every condition has every seed; FB models are shared by the three goals of a layout
    cells = Counter((j.labels["env"], j.labels["layout"], j.labels["task"]) for j in _jobs(core, "fpg"))
    assert set(cells.values()) == {3} and len(cells) == 9
    for j in _jobs(core, "adapt"):
        fb = next(f for f in _jobs(core, "fb_pretrain") if f.job_id == j.identity["fb_job"])
        assert (fb.labels["env"], fb.labels["layout"], fb.labels["seed"]) == (j.labels["env"], j.labels["layout"], j.labels["seed"])
    # budgets: the spec's ceilings
    f = _jobs(core, "fpg", env="finite")[0].identity
    c = _jobs(core, "fpg", env="continuous")[0].identity
    assert f["budget_transitions"] == 50_000 and c["budget_transitions"] == 100_000
    fb_f = _jobs(core, "fb_pretrain", env="finite")[0].identity
    fb_c = _jobs(core, "fb_pretrain", env="continuous")[0].identity
    assert fb_f["fb"]["n_updates"] == 10_000 and fb_c["fb"]["n_updates"] == 20_000
    items = ex.data_items(core)
    assert items["replay/finite/four_rooms/s0"].spec["n_transitions"] == 50_000
    assert items["replay/continuous/four_rooms/s0"].spec["n_transitions"] == 100_000
    assert all(i.spec["n_trajectories"] == 4 for i in items.values() if i.kind == "demos")
    assert _jobs(core, "adapt")[0].identity["dm"]["steps"] <= 300


def test_ablations_and_reduced_core_reuse_the_core_jobs(suites):
    core = {j.job_id for j in ex.expand_jobs(suites["core"])}
    abl = ex.expand_jobs(suites["compact_ablations"])
    extra = [j for j in abl if j.job_id not in core]
    fpg_extra = [j for j in extra if j.type == "fpg"]
    fb_extra = [j for j in extra if j.type == "fb_pretrain"]
    assert len(fpg_extra) == 6 and {j.labels["divergence"] for j in fpg_extra} == {"kl_dq", "kl_qd"}
    assert all((j.labels["layout"], j.labels["task"]) == ("four_rooms", "goal_5_5") for j in fpg_extra)
    assert len(fb_extra) == 3 and all(j.identity["fb"]["orth_coef"] == 0.0 for j in fb_extra)
    base_js = [j for j in abl if j.type == "fpg" and j.labels["divergence"] == "js"]
    assert len(base_js) == 3 and all(j.job_id in core for j in base_js)
    red = ex.expand_jobs(suites["reduced"])
    assert {j.job_id for j in red} <= core
    conds = {(j.labels["env"], j.labels["layout"], j.labels["task"]) for j in red if j.type == "fpg"}
    assert conds == {("finite", "open", "goal_5_5"), ("finite", "four_rooms", "goal_5_5"),
                     ("finite", "four_rooms", "goal_5_1"), ("continuous", "four_rooms", "goal_5_5")}
    assert suites["reduced"].seeds == [0, 1, 2]


def test_pilot_uses_its_own_seed_data_and_a_small_declared_grid(suites):
    pilot, core = suites["pilot"], suites["core"]
    assert pilot.seeds == [999] and pilot.data_id != core.data_id and pilot.reference_seed != core.reference_seed
    lrs = {env: {j.identity["fpg"]["lr"] for j in _jobs(pilot, "fpg", env=env) if j.identity["fpg"].get("density", "kde") == "kde"}
           for env in ("finite", "continuous")}
    assert len(lrs["finite"]) <= 3 and len(lrs["continuous"]) <= 3
    taus = {j.identity["fb"]["tau"] for j in _jobs(pilot, "fb_pretrain")}
    assert taus == {0.1, 0.3, 1.0}
    dens = {j.identity["fpg"]["density"] for j in _jobs(pilot, "fpg", env="continuous")}
    assert dens == {"kde", "histogram"}
    base_pilot = next(j for j in _jobs(pilot, "fpg", env="finite", fpg_variant="base"))
    base_core = _jobs(core, "fpg", env="finite")[0]
    assert base_pilot.identity["budget_transitions"] == base_core.identity["budget_transitions"]
    assert pilot.pilot["overhead"] >= 0.5


def test_job_identifiers_are_deterministic_hashes_of_the_identity(suites):
    for suite in suites.values():
        first = ex.expand_jobs(suite)
        second = ex.expand_jobs(ex.load_suite_config(ROOT / suite.path))
        assert [j.job_id for j in first] == [j.job_id for j in second]
        for j in first:
            h = sha256_json(j.identity)
            assert j.identity_sha256 == h and j.job_id.endswith(h[:10])
        assert len({j.job_id for j in first}) == len(first)


def test_random_streams_are_independent_across_tasks_and_paired_across_variants(suites):
    core = suites["core"]
    keys = Counter((j.identity["seed"], tuple(j.identity["stream"])) for j in ex.expand_jobs(core) if j.type in ("fpg", "fb_pretrain", "adapt"))
    assert max(keys.values()) == 1, "two training jobs of the core share a seed and a stream"
    abl = suites["compact_ablations"]
    by_seed = {}
    for j in _jobs(abl, "fpg"):
        by_seed.setdefault(j.labels["seed"], set()).add(tuple(j.identity["stream"]))
    assert all(len(s) == 1 for s in by_seed.values()), "divergence variants of one seed must share their stream (CRN)"
    items = ex.data_items(core)
    pairs = Counter((i.spec["seed"], tuple(i.spec["stream"])) for i in items.values())
    assert max(pairs.values()) == 1
    train_streams = {tuple(j.identity["stream"][:1]) for j in ex.expand_jobs(core) if j.type in ("fpg", "fb_pretrain", "adapt")}
    eval_streams = {tuple(j.identity["stream"][:1]) for j in ex.expand_jobs(core) if j.type in ("eval", "eval_expert")}
    assert eval_streams == {("evaluation",)} and not (train_streams & eval_streams)


def _write(tmp_path, raw, name="s.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


@pytest.fixture
def smoke_raw():
    raw = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    raw["environment_config"] = str(ROOT / "configs" / "environment.yaml")
    return raw


@pytest.mark.parametrize("mutation", [
    "unknown_top_key", "seeds_not_kind", "explicit_seed_list", "continuous_only_finite", "budget_not_multiple",
    "duplicate_variant", "unknown_goal", "continuous_open_layout", "overrun_dm_steps", "low_convergence",
])
def test_strict_schema_rejects_mistakes(tmp_path, smoke_raw, mutation):
    raw = smoke_raw
    if mutation == "unknown_top_key":
        raw["budgets"] = {}
    elif mutation == "seeds_not_kind":
        raw["seeds"] = "final"
    elif mutation == "explicit_seed_list":
        raw["seeds"] = [0, 1]
    elif mutation == "continuous_only_finite":
        raw["methods"]["fpg"]["finite"]["settings"]["density"] = "kde"
    elif mutation == "budget_not_multiple":
        raw["methods"]["fpg"]["finite"]["budget_transitions"] = 1000
    elif mutation == "duplicate_variant":
        raw["methods"]["fb"]["variants"] = {"base": {}, "same": {"all": {"settings": {"tau": 0.1}}}}
    elif mutation == "unknown_goal":
        raw["grid"][0]["tasks"] = ["goal_9_9"]
    elif mutation == "continuous_open_layout":
        raw["grid"][1]["layout"] = "open"
    elif mutation == "overrun_dm_steps":
        raw["methods"]["adaptation"]["dm_snapshot_steps"] = [0, 500]
    elif mutation == "low_convergence":
        raw["evaluation"]["convergence"]["js_max_nats"] = 0.9
    with pytest.raises((ex.ConfigError, ValueError)):
        ex.load_suite_config(_write(tmp_path, raw))


def test_seed_manifest_needs_distinct_seeds(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "seeds.yaml").read_text(encoding="utf-8"))
    assert ex.load_seed_manifest(ROOT / "configs" / "seeds.yaml").final_seeds == (0, 1, 2)
    raw["pilot_seed"] = 0
    with pytest.raises(ex.ConfigError):
        ex.load_seed_manifest(_write(tmp_path, raw, "seeds.yaml"))


def test_inheritance_merges_mappings_and_replaces_lists_and_variants():
    base = {"a": {"x": 1, "y": [1, 2]}, "variants": {"base": {}, "v": {}}, "keep": 3}
    over = {"a": {"y": [3]}, "variants": {"base": {}}}
    out = ex.deep_merge(base, over)
    assert out == {"a": {"x": 1, "y": [3]}, "variants": {"base": {}}, "keep": 3}
    assert base["a"]["y"] == [1, 2]
