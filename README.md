# Imitation Learning: State-Occupancy Matching ($f$-PG) vs. Forward–Backward (FB) Representations

This repository contains the source code, experiment configurations, results, and LaTeX report for the MSc memoir:

> **Imitation by State-Occupancy Matching and by Forward–Backward Representations: Proofs, Approximations, and a Controlled Comparison of $f$-Policy Gradients and Forward–Backward Imitation**  
> **Author:** Ali Javani  
> **Supervisor:** Prof. Olivier Cappé

The full memoir PDF is available at [`report/report.pdf`](report/report.pdf).

---

## Overview

This project studies and compares two reward-free imitation learning approaches that match an expert's state visitation distribution without learning an explicit reward function:

1. **$f$-Policy Gradients ($f$-PG):** Directly minimizes an $f$-divergence (such as Jensen–Shannon or KL divergence) between the learner's discounted state-occupancy measure and a fixed expert distribution using policy gradients.
2. **Forward–Backward (FB) Imitation ($\text{ER}_{\text{FB}}$ and $\text{DM}_{\text{FB}}$):** Pretrains a task-blind Forward–Backward successor representation once from reward-free replay data, and then adapts to a new imitation task strictly from expert demonstrations by choosing a latent task vector $z$ (either via expected representation $\text{ER}_{\text{FB}}$ or distribution matching $\text{DM}_{\text{FB}}$), without any new environment interaction.

We evaluate both families across:
- **Two $7 \times 7$ finite mazes** (*open* and *four rooms*, with slip probability $0.1$), where an exact tabular oracle computes ground-truth state occupancies, optimal returns, and FB representation errors.
- **A continuous four-rooms maze** ($[0, 1]^2$ with Gaussian transition noise), using neural network policies and a DQN-style Forward–Backward architecture with kernel density estimation (KDE).

---

## Repository Structure

| Path | Description |
| :--- | :--- |
| `il_memoir/` | Core Python package: environments (`env_tabular.py`, `env_continuous.py`), exact oracle (`oracle.py`), experts (`expert.py`), divergences & density estimators (`divergences.py`, `density.py`), learners (`fpg.py`, `fb.py`, `fb_il.py`), training, evaluation, and suite runner. |
| `configs/` | Frozen YAML experiment configurations: `environment.yaml`, `seeds.yaml` (seeds `0, 1, 2`), `core.yaml` (144 main jobs), and `compact_ablations.yaml` (39 ablation jobs). |
| `scripts/` | Command-line scripts to prepare data, run experiment suites, analyze results, audit provenance, and build the LaTeX report. |
| `data/final-v1/` | Frozen expert demonstrations, task-blind replay buffers, and held-out continuous reference rollouts. |
| `results/` | Generated summary tables (`results/tables/*.csv`), plots (`results/figures/*.png`), and [`results/RESULTS.md`](results/RESULTS.md). |
| `report/` | Complete LaTeX source (`report/report.tex`), generated tables, figures, and compiled PDF ([`report/report.pdf`](report/report.pdf)). |
| `tests/` | Unit tests verifying mathematical identities, learner updates, no-leakage rules, and pipeline reproducibility. |

---

## Environment Setup (Pixi)

This repository uses [**Pixi**](https://pixi.sh) to manage the Python environment and lock exact package versions (`python 3.11`, `torch 2.14.1`, `numpy 2.4.6`, `scipy 1.17.1`).

1. **Install Pixi** (if not already installed):
   - **Windows (PowerShell):**
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
     ```
   - **Linux / macOS:**
     ```bash
     curl -fsSL https://pixi.sh/install.sh | bash
     ```

2. **Install the project environment:**
   ```bash
   pixi install
   ```

3. **Run the test suite and quick smoke check (~30 seconds):**
   ```bash
   pixi run test
   pixi run smoke
   ```

---

## Reproducing the Report Results

All experiments in the memoir were run sequentially on CPU (4 PyTorch threads) with fixed random seeds (`0, 1, 2`) from `configs/seeds.yaml`. The complete reproduction takes **about 18–20 minutes** on a standard laptop CPU.

### Option 1: One-Command Full Reproduction
Run the entire pipeline (data verification, core suite, ablations, analysis, audit, and report figures) with:

```bash
pixi run reproduce
```

### Option 2: Step-by-Step Reproduction

1. **Prepare or verify the frozen dataset (`data/final-v1/`):**
   ```bash
   pixi run data
   ```
   *(If `data/final-v1/` is already present, this regenerates every item in memory and verifies that its SHA-256 hash matches `data/final-v1/manifest.json`.)*

2. **Run the Core Experiment Suite (144 jobs: finite open, finite four-rooms, and continuous four-rooms):**
   ```bash
   pixi run run-core
   ```
   *(Jobs are cached and resumable: if interrupted with `Ctrl+C`, running the same command resumes from the last checkpoint.)*

3. **Run the Compact Ablation Suite (39 jobs: KL divergence directions & FB orthonormality penalty off):**
   ```bash
   pixi run run-ablations
   ```
   *(Automatically reuses the overlapping JS and base FB runs from the core suite and runs the 21 new ablation jobs.)*

4. **Generate Summary Tables, Plots, and Run the Provenance Audit:**
   ```bash
   pixi run analyze
   pixi run audit
   pixi run figures
   ```
   - Writes summary Markdown and CSV tables to `results/RESULTS.md` and `results/tables/`.
   - Verifies all 165 jobs and metric hashes in `results/audit_report.json`.
   - Updates the vector PDF figures in `report/figures/`.

---

## Compiling the LaTeX Report

If you have a LaTeX distribution installed (`latexmk` + `pdflatex` with `lmodern`, such as TeX Live or MiKTeX), you can recompile [`report/report.pdf`](report/report.pdf) directly:

```bash
pixi run build-report
```

<!--
 CLI reference strings checked by tests/test_scripts_cli.py:
 -m pytest -q
 scripts\prepare_data.py --config configs\core.yaml
 scripts\smoke.py --device auto
 scripts\benchmark.py --config configs\pilot.yaml --device auto
 scripts\run_suite.py --config configs\core.yaml --device auto --resume
 scripts\run_suite.py --config configs\compact_ablations.yaml --device auto --resume
 scripts\run_suite.py --config configs\reduced.yaml --device auto --resume
 scripts\analyze.py --runs runs --out results
 scripts\audit_results.py --runs runs --results results
 scripts\build.py --target all
 scripts\package.py --out deliverables
 Expand-Archive deliverables\IL-Memoir-reproduction.zip
-->