"""Imitation-learning memoir code: f-PG versus FB imitation on small mazes.

Modules written in Prompt 3A (environments, oracle, experts, divergences and
density estimators):

- ``config``        load and validate ``configs/environment.yaml``;
- ``common``        reproducible random streams, trajectory containers, action sampling;
- ``env_tabular``   finite 7x7 layouts, the learner-facing simulator and the
                    oracle-only export of the transition matrix P;
- ``env_continuous`` continuous maze in [0,1]^2 with segment collision checks;
- ``tasks``         goal geometry and the evaluation reward (expert/evaluation only);
- ``oracle``        exact finite occupancies, successor measures, policy
                    evaluation, value iteration and backward induction;
- ``expert``        finite oracle expert and the continuous route controller;
- ``divergences``   f-divergence generators, smoothing and the f-PG signal factor;
- ``density``       discount weights, finite counts, free-space histogram and
                    weighted kernel density estimates.

Learner modules written in Prompt 3B (they never import ``oracle``, ``tasks``,
``expert`` or ``evaluation``; ``tests/test_no_leakage.py`` checks this):

- ``nets``          C^1 MLPs with seeded initialization, observation encoding,
                    checkpoint helpers (shared by the learners);
- ``fpg``           f-PG with a fixed expert target: policies, cross-fitted
                    plug-in signal, signal-to-go, baseline, gradient estimator,
                    optional PPO surrogate, the resumable ``FPGLearner``;
- ``fb``            task-blind replay, FB networks, successor loss,
                    orthonormality penalty, Polyak targets, Gram matrix, the
                    resumable ``FBPretrainer`` and the frozen ``FBModel``;
- ``fb_il``         ER_FB and DM_FB on a frozen ``FBModel`` (no simulator access).

Runners, evaluation, analysis and reproducibility, written in Prompt 3C:

- ``experiment``    suite configurations, the frozen seed manifest and the
                    deterministic job graph (identifiers hash every setting);
- ``datasets``      frozen data on disk: manifest, content hashes, loaders (the
                    learner-side loader returns demonstration states only);
- ``prepare``       generation of demonstrations, task-blind replay (coverage
                    checked) and held-out references; never overwrites other data;
- ``train``         f-PG, FB pretraining and ER/DM adaptation jobs with periodic
                    checkpoints, Ctrl+C handling and exact resume;
- ``evaluation``    held-out metrics common to every method; the only module
                    besides the expert that calls the oracle;
- ``runner``        sequential, cached, resumable suites with manifests, stale-code
                    detection and invalidation;
- ``logging``       run identity, code scopes, provenance, atomic writes, statuses
                    (``il_memoir.logging``; the standard library is unaffected);
- ``hardware``      device policy (pilot choice, frozen suite device, Slurm),
                    hardware records, synchronized timers, memory peaks;
- ``analysis``, ``audit``  RESULTS.md, tables, figures and provenance from saved
                    runs; integrity audit;
- ``pilot``         CPU/GPU timing, device choice, learning pilot and ETA;
- ``checks``        reload and resume comparisons (smoke checks);
- ``documents``, ``packaging``  LaTeX build, Overleaf bundles, deliverables.

Importing this package does not import any submodule.
"""

__version__ = "0.5.0"
