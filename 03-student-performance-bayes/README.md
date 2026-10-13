# Student Performance Prediction with Bayesian Inference

An educational project designed to predict student performance (pass/fail and final grades) using Bayesian Networks and teach core probabilistic graphical modeling concepts from scratch.

---

## Key Concepts and Objectives

1. **Bayes' Theorem and Conditional Independence**: Understanding prior, likelihood, posterior, and d-separation structures (chains, forks, colliders / explaining away).
2. **Exact Inference**:
   - Enumeration
   - Variable Elimination (VE) with heuristic elimination orderings
3. **Approximate Inference**:
   - Rejection Sampling
   - Likelihood Weighting
   - Gibbs Sampling (MCMC)
4. **Parameter Learning**:
   - Maximum Likelihood Estimation (MLE)
   - Bayesian Estimation with Dirichlet / Laplace smoothing
5. **Empirical Evaluation**:
   - Comparing Exact vs. Approximate inference (runtime, error, KL divergence)
   - Baseline benchmarks (Naive Bayes, Logistic Regression)

---

## Project Structure

```text
03-student-performance-bayes/
├── AGENTS.md
├── configs/
│   ├── discretization.yaml
│   └── network.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── DECISIONS.md
├── notebooks/
│   └── 01_bayes_theorem.ipynb
├── scripts/
│   └── download_data.py
├── src/
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

### 1. Environment Setup

Using `uv` or `pip`:

```bash
cd 03-student-performance-bayes
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or with `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data

```bash
python scripts/download_data.py
```

### 3. Run Notebooks

```bash
jupyter lab notebooks/
```
