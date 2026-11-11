"""Learner-safe shared helpers.

Nothing in this module touches a reward, a goal or a transition matrix.  It provides

- ``make_rng``: independent, reproducible random streams derived from one seed
  and a tuple of stream names (``numpy.random.SeedSequence`` spawn keys), so
  that, for example, demonstration sampling and evaluation rollouts never share
  random numbers; ``stream_labels`` copies the labels for a record without
  changing their types;
- ``Trajectories``: a container for ``n`` trajectories with ``H`` transitions
  (states ``s_0..s_H`` and actions ``a_0..a_{H-1}``; action ``a_t`` is taken in
  ``s_t`` and leads to ``s_{t+1}``, NOTATION section 1);
- ``Demonstrations``: expert trajectories whose learner view is the states only;
- ``sample_actions`` and ``TabularPolicy``: sampling from a batch of action
  distributions (float32 network outputs are accepted).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Union

import numpy as np

PROB_ATOL = 1e-9          # row-sum tolerance for float64 probabilities
PROB_ATOL_LOW = 1e-5      # row-sum tolerance for float32/float16 probabilities (network outputs)


def _stream_word(item: object) -> int:
    """Map a stream label to a non-negative integer, identically on every platform."""
    if isinstance(item, (bool, np.bool_)):
        raise TypeError("stream labels must be str or non-negative int, not bool")
    if isinstance(item, (int, np.integer)):
        value = int(item)
        if value < 0:
            raise ValueError("integer stream labels must be non-negative")
        return value
    if isinstance(item, str):
        return int.from_bytes(hashlib.sha256(item.encode("utf-8")).digest()[:8], "little")
    raise TypeError(f"stream labels must be str or non-negative int, got {type(item).__name__}")


def make_rng(seed: int, *stream: Union[str, int]) -> np.random.Generator:
    """Return a PCG64 generator for ``seed`` and the stream labels ``stream``.

    The same ``(seed, stream)`` always gives the same sequence.  Different
    labels give different SeedSequence spawn keys, i.e. statistically
    independent streams in the sense of NumPy's ``SeedSequence.spawn``.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an int")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    seq = np.random.SeedSequence(entropy=int(seed), spawn_key=tuple(_stream_word(s) for s in stream))
    return np.random.Generator(np.random.PCG64(seq))


def stream_labels(labels: Sequence[Union[str, int]]) -> List[Union[str, int]]:
    """Copy of stream labels for a record, with their types kept: ``str`` stays ``str``, integers become ``int``.

    ``make_rng`` maps the integer ``2`` and the string ``"2"`` to different
    streams, so a record must not stringify labels.  The returned list is
    JSON-safe (NumPy integers become ``int``), and ``make_rng(seed, *record)``
    is the same stream as ``make_rng(seed, *labels)``, also after a JSON round
    trip.  Labels are validated as in ``make_rng``.
    """
    if isinstance(labels, (str, bytes)):
        raise TypeError("stream labels must be a sequence of labels, not a single string")
    record: List[Union[str, int]] = []
    for item in labels:
        _stream_word(item)
        record.append(str(item) if isinstance(item, str) else int(item))
    return record


def check_probability_rows(probs: np.ndarray, atol: float = PROB_ATOL) -> np.ndarray:
    """Validate a 2-D array of probability rows; return a float64 copy with rows renormalized.

    The row-sum tolerance is ``atol`` for float64 input and ``PROB_ATOL_LOW``
    for lower-precision input (a float32 softmax over five actions has row-sum
    errors of order 1e-7).
    """
    raw = np.asarray(probs)
    tol = atol if raw.dtype == np.float64 else max(atol, PROB_ATOL_LOW)
    probs = np.array(raw, dtype=np.float64, copy=True)
    if probs.ndim != 2:
        raise ValueError(f"expected a 2-D array of action probabilities, got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("action probabilities must be finite")
    if np.any(probs < -tol):
        raise ValueError("action probabilities must be non-negative")
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=tol):
        raise ValueError("action probability rows must sum to one")
    probs = np.clip(probs, 0.0, None)
    return probs / probs.sum(axis=1, keepdims=True)


def sample_actions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one action per row of ``probs`` by inversion of the cumulative sums.

    Actions with probability zero are never drawn: if ``probs[k] = 0`` the
    cumulative sum does not increase at ``k``, so the first index with
    ``u < cdf`` is never ``k``.
    """
    probs = check_probability_rows(probs)
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(probs.shape[0])
    return np.argmax(u[:, None] < cdf, axis=1).astype(np.int64)


class TabularPolicy:
    """Stationary Markov policy on a finite state space, given by a matrix ``pi[s, a]`` (copied)."""

    def __init__(self, pi: np.ndarray):
        self.pi = check_probability_rows(pi)
        self.pi.setflags(write=False)

    @property
    def n_states(self) -> int:
        return self.pi.shape[0]

    @property
    def n_actions(self) -> int:
        return self.pi.shape[1]

    def __call__(self, states: np.ndarray) -> np.ndarray:
        return self.pi[np.asarray(states, dtype=np.int64)]


PolicyLike = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]


def as_policy(policy: PolicyLike) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable ``states -> (n, A) probabilities``; a matrix becomes a TabularPolicy."""
    if isinstance(policy, np.ndarray):
        return TabularPolicy(policy)
    if callable(policy):
        return policy
    raise TypeError("policy must be an (S, A) matrix or a callable returning action probabilities")


@dataclass(frozen=True)
class Trajectories:
    """``n`` trajectories of ``H`` transitions.

    ``states`` has shape ``(n, H+1)`` (finite indices, int64) or ``(n, H+1, 2)``
    (continuous positions, float64); ``actions`` has shape ``(n, H)``.
    """

    states: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        states = np.asarray(self.states)
        actions = np.asarray(self.actions, dtype=np.int64)
        if states.ndim not in (2, 3):
            raise ValueError(f"states must have shape (n, H+1) or (n, H+1, 2), got {states.shape}")
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape (n, H), got {actions.shape}")
        if states.shape[0] != actions.shape[0] or states.shape[1] != actions.shape[1] + 1:
            raise ValueError(f"inconsistent shapes: states {states.shape}, actions {actions.shape}")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)

    @property
    def n_traj(self) -> int:
        return int(self.actions.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])

    @property
    def next_states(self) -> np.ndarray:
        """``s_1..s_H`` for every trajectory: the states weighted by the occupancy estimators."""
        return self.states[:, 1:]


@dataclass(frozen=True)
class Demonstrations:
    """Expert trajectories.  Learners receive ``learner_states()`` only (METHOD_MAP section 1).

    ``trajectories.actions`` are the expert's actions, kept for diagnostics
    (state-action occupancy) and never handed to the main learners; runners
    pass learners the array returned by :meth:`learner_states`, not this object.
    """

    trajectories: Trajectories
    meta: Dict[str, Any] = field(default_factory=dict)

    def learner_states(self) -> np.ndarray:
        """Copy of the states ``s_0..s_H`` of every demonstration (no actions, no reward)."""
        return np.array(self.trajectories.states, copy=True)
