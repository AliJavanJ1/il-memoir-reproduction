"""Configuration values against the spec, strict validation, random streams and action sampling."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from il_memoir.common import TabularPolicy, Trajectories, make_rng, sample_actions, stream_labels
from il_memoir.config import DEFAULT_ENV_CONFIG, environment_config_from_dict
from il_memoir.env_tabular import four_rooms_rule_blocked


def test_config_matches_spec(cfg):
    assert cfg.gamma == 0.95 and cfg.horizon == 100 and cfg.grid_size == 7
    assert cfg.start == (1, 1)
    assert dict(cfg.goals) == {"goal_5_1": (5, 1), "goal_5_5": (5, 5), "goal_1_5": (1, 5)}
    assert cfg.blocked_cells("open") == ()
    # spec section 5: walls x=3 except y=1,5; y=3 except x=1,5; intersection blocked
    assert set(cfg.blocked_cells("four_rooms")) == set(four_rooms_rule_blocked(7, 3, (1, 5)))
    assert (3, 3) in cfg.blocked_cells("four_rooms")
    assert cfg.action_names == ("stay", "right", "up", "left", "down")
    assert cfg.action_deltas == ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))
    assert cfg.slip_prob == 0.1
    assert cfg.continuous_layout == "four_rooms"
    assert np.isclose(cfg.step_length, 1 / 7) and cfg.noise_std == 0.01
    assert np.isclose(cfg.reward_width, 1 / 14) and np.isclose(cfg.goal_radius, 1 / 14)
    assert cfg.success_rule == "hit"
    assert cfg.kde_bandwidth == 0.05 and cfg.histogram_bins == 20 and cfg.smoothing_eps == 0.01


def _raw():
    with open(DEFAULT_ENV_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update({"unexpected": 1}),
        lambda d: d["finite"].update({"slip_probability": 0.1}),
        lambda d: d.update({"gamma": 1.0}),
        lambda d: d.update({"horizon": 0}),
        lambda d: d["grid"]["layouts"]["four_rooms"].append([1, 1]),     # start cell blocked
        lambda d: d["grid"]["layouts"]["four_rooms"].append([3, 0]),     # duplicate wall
        lambda d: d["actions"].append({"name": "jump", "dx": 2, "dy": 0}),
        lambda d: d["finite"].update({"slip_prob": 1.0}),
        lambda d: d["task"].update({"success": "sometimes"}),
        lambda d: d["density"].update({"smoothing_eps": -0.1}),
        lambda d: d["density"].update({"smoothing_eps": 0.0}),                    # eps must be > 0
        lambda d: d.update({"schema": "other"}),
        lambda d: d.update({"actions": [a for a in d["actions"] if a["name"] != "stay"]}),
        lambda d: d["grid"]["goals"].update({"duplicate": [5, 5]}),
        lambda d: d["grid"].update({"start": [True, 1]}),
        lambda d: d["grid"].update({"start": [1.0, 1]}),
        lambda d: d["grid"]["layouts"]["four_rooms"].extend([[1, 0], [0, 1]]),   # (0,0) cut off from the start
        # non-finite numbers (YAML .nan / .inf); every comparison with NaN is false, so range checks alone pass them
        lambda d: d.update({"gamma": float("nan")}),
        lambda d: d["finite"].update({"slip_prob": float("nan")}),
        lambda d: d["continuous"].update({"step_cells": float("nan")}),
        lambda d: d["continuous"].update({"step_cells": float("inf")}),
        lambda d: d["continuous"].update({"noise_std": float("nan")}),
        lambda d: d["continuous"].update({"noise_std": float("inf")}),
        lambda d: d["task"].update({"reward_width_cells": float("nan")}),
        lambda d: d["task"].update({"goal_radius_cells": float("inf")}),
        lambda d: d["density"].update({"kde_bandwidth": float("nan")}),
        lambda d: d["density"].update({"kde_bandwidth": float("inf")}),
        lambda d: d["oracle"].update({"value_iteration_tol": float("nan")}),
        lambda d: d["oracle"].update({"tie_tol": float("nan")}),
        lambda d: d["oracle"].update({"tie_tol": float("inf")}),
        # wrong types that were converted silently
        lambda d: d.update({"gamma": "0.95"}),
        lambda d: d.update({"horizon": True}),
        lambda d: d["density"].update({"histogram_bins": 20.5}),
        lambda d: d["continuous"].update({"max_noise_redraws": 1000.7}),
        lambda d: d["actions"][1].update({"dx": 1.5}),
    ],
)
def test_config_rejects_invalid(mutate):
    raw = copy.deepcopy(_raw())
    mutate(raw)
    with pytest.raises(ValueError):
        environment_config_from_dict(raw)


def test_config_hash_is_stable_and_sensitive(cfg):
    again = environment_config_from_dict(_raw())
    assert again.sha256() == cfg.sha256()
    raw = _raw()
    raw["finite"]["slip_prob"] = 0.2
    assert environment_config_from_dict(raw).sha256() != cfg.sha256()


def test_learner_view_omits_goals_reward_and_oracle(cfg):
    view = cfg.learner_dict()
    text = repr(view).lower()
    assert "goal" not in text and "reward" not in text and "success" not in text
    assert "task" not in view and "oracle" not in view and "goals" not in view["grid"]
    assert "slip" not in text and "noise" not in text and "finite" not in view   # transition parameters define P
    assert view["continuous"] == {"layout": cfg.continuous_layout}
    assert view["gamma"] == cfg.gamma and view["horizon"] == cfg.horizon
    assert view["density"]["kde_bandwidth"] == cfg.kde_bandwidth


def test_make_rng_reproducible_and_separated():
    a1 = make_rng(7, "demos", "four_rooms", 3).random(5)
    a2 = make_rng(7, "demos", "four_rooms", 3).random(5)
    b = make_rng(7, "eval", "four_rooms", 3).random(5)
    c = make_rng(8, "demos", "four_rooms", 3).random(5)
    assert np.array_equal(a1, a2)
    assert not np.allclose(a1, b) and not np.allclose(a1, c)
    with pytest.raises(ValueError):
        make_rng(-1)
    with pytest.raises(TypeError):
        make_rng(1, 2.5)


def test_stream_labels_keep_their_types_for_records():
    """Regression (external 3A review): the integer 2 and the string "2" are different streams."""
    import json

    labels = ("demos", np.int64(3), 4)
    record = json.loads(json.dumps(stream_labels(labels)))
    assert record == ["demos", 3, 4] and [type(v) for v in record] == [str, int, int]
    assert np.array_equal(make_rng(5, *record).random(4), make_rng(5, *labels).random(4))
    assert not np.array_equal(make_rng(5, "demos", "3", 4).random(4), make_rng(5, *labels).random(4))
    for bad in [("a", True), ("a", -1), ("a", 1.5), "abc"]:
        with pytest.raises((TypeError, ValueError)):
            stream_labels(bad)


def test_sample_actions_frequencies():
    rng = make_rng(11, "sample_actions")
    probs = np.tile([0.1, 0.2, 0.7, 0.0, 0.0], (200_000, 1))
    acts = sample_actions(probs, rng)
    freq = np.bincount(acts, minlength=5) / acts.size
    se = np.sqrt(probs[0] * (1 - probs[0]) / acts.size)
    assert np.all(np.abs(freq - probs[0]) <= 5 * se + 1e-12)
    assert freq[3] == 0.0 and freq[4] == 0.0
    with pytest.raises(ValueError):
        sample_actions(np.array([[0.5, 0.6]]), rng)


def test_sample_actions_accepts_float32_network_outputs():
    """A float32 softmax has row-sum errors of order 1e-7; it must be accepted (3B networks are float32)."""
    rng = make_rng(12, "float32")
    logits = rng.normal(size=(50_000, 5)).astype(np.float32) * 3
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs32 = e / e.sum(axis=1, keepdims=True)
    assert probs32.dtype == np.float32 and np.max(np.abs(probs32.sum(axis=1) - 1)) > 1e-9
    acts = sample_actions(probs32, rng)
    assert acts.shape == (50_000,) and acts.min() >= 0 and acts.max() <= 4
    with pytest.raises(ValueError):
        sample_actions(np.array([[0.5, 0.6]], dtype=np.float32), rng)


def test_tabular_policy_copies_the_matrix():
    pi = np.full((3, 2), 0.5)
    pol = TabularPolicy(pi)
    pi[0, 0] = 0.9                       # the caller's array stays writable and independent
    assert pol(np.array([0]))[0, 0] == 0.5


def test_trajectories_validate_shapes():
    Trajectories(np.zeros((2, 4), dtype=np.int64), np.zeros((2, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        Trajectories(np.zeros((2, 4)), np.zeros((2, 4)))
