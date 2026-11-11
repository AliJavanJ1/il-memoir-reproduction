"""Shared fixtures.  All randomness comes from fixed seeds through ``il_memoir.common.make_rng``."""
from __future__ import annotations

import pytest

from il_memoir.config import load_environment_config
from il_memoir.env_continuous import make_continuous_maze
from il_memoir.env_tabular import make_grid_maze, transition_matrix
from il_memoir.tasks import make_tasks

TEST_SEED = 20261005


@pytest.fixture(scope="session")
def cfg():
    return load_environment_config()


@pytest.fixture(scope="session")
def four_rooms(cfg):
    return make_grid_maze(cfg, "four_rooms")


@pytest.fixture(scope="session")
def open_maze(cfg):
    return make_grid_maze(cfg, "open")


@pytest.fixture(scope="session")
def four_rooms_P(four_rooms):
    return transition_matrix(four_rooms)


@pytest.fixture(scope="session")
def cont_env(cfg):
    return make_continuous_maze(cfg)


@pytest.fixture(scope="session")
def tasks(cfg):
    return make_tasks(cfg)


# ----------------------------------------------------------------------------- Prompt 3C fixtures
# TEST FIXTURES: tiny suites (configs/smoke.yaml, tests/fixtures/runner_tiny.yaml) run in temporary
# directories.  Their outputs are execution checks, never results.
def _run_suite(config, base, **kwargs):
    from il_memoir.experiment import load_suite_config
    from il_memoir.logging import StopController, read_json
    from il_memoir.runner import Runner

    suite = load_suite_config(config)
    runner = Runner(suite, base / "runs", base / "data", device=kwargs.pop("device", "cpu"),
                    stop=kwargs.pop("stop", None) or StopController(), log=kwargs.pop("log", lambda m: None), **kwargs)
    code = runner.run(resume=True)
    man_path = base / "runs" / "suites" / suite.name / "manifest.json"
    return {"code": code, "suite": suite, "runs": base / "runs", "data": base / "data", "base": base,
            "manifest": read_json(man_path) if man_path.exists() else None}


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory):
    """The smoke suite (both environments, every job type), completed once per session (TEST FIXTURE)."""
    out = _run_suite("configs/smoke.yaml", tmp_path_factory.mktemp("smoke_fixture"))
    assert out["code"] == 0
    return out


@pytest.fixture(scope="session")
def runner_run(tmp_path_factory):
    """The finite-only runner fixture suite, completed once per session (TEST FIXTURE)."""
    out = _run_suite("tests/fixtures/runner_tiny.yaml", tmp_path_factory.mktemp("runner_fixture"))
    assert out["code"] == 0
    return out


@pytest.fixture
def run_suite():
    return _run_suite
