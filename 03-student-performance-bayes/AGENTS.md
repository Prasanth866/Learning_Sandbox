# AGENTS.md — Student Performance Prediction with Bayesian Inference

> Instructions for the coding agent (Claude Code / Cursor / Copilot agent) building this project.
> Read this whole file before writing any code. Follow the phases in order.

---

## 1. Project Goal

Build an **educational, runnable project** that predicts student performance (final grade / pass-fail) using a **Bayesian Network**, and that **teaches the concepts** of:

1. Bayes' theorem and conditional independence
2. **Exact inference**: enumeration, variable elimination
3. **Approximate inference**: rejection sampling, likelihood weighting, Gibbs sampling (MCMC)
4. Parameter learning (MLE vs. Bayesian / Dirichlet priors)
5. Comparing exact vs. approximate results (accuracy, error, runtime, convergence)
6. Comparing the Bayesian Network against baselines (Naive Bayes, Logistic Regression)

The project is a **learning tool first, a predictor second**. Clarity of code and explanation beats raw accuracy. Every algorithm must be implemented **from scratch** (NumPy only) *and* cross-checked against a library (`pgmpy`).

---

## 2. Core Concepts the Project Must Explain

The README, notebooks and docstrings must explain each of these in plain language with a small example.

### 2.1 Bayes' theorem
```
P(H | E) = P(E | H) · P(H) / P(E)
```
Example: P(Fail | LowAttendance) from P(LowAttendance | Fail), P(Fail), P(LowAttendance).

### 2.2 Bayesian Network (BN)
A directed acyclic graph where each node has a conditional probability table (CPT) given its parents. The joint distribution factorizes as:
```
P(X1, ..., Xn) = Π_i P(Xi | Parents(Xi))
```
This factorization is what makes inference tractable.

### 2.3 Conditional independence & d-separation
Explain chain, fork and collider structures with one student-domain example each:
- Chain: `StudyHours → Understanding → ExamScore`
- Fork: `ParentalSupport → StudyHours`, `ParentalSupport → Internet`
- Collider: `Motivation → Understanding ← Teaching` (explaining away)

### 2.4 Exact inference
Compute `P(Query | Evidence)` precisely.
- **Enumeration**: sum the full joint over hidden variables. Simple, exponential.
- **Variable Elimination (VE)**: push sums inside products, eliminate hidden variables one at a time using factor product and marginalization. Show how **elimination order** changes cost.

### 2.5 Approximate inference
Used when exact inference is too expensive. All are Monte Carlo methods.
- **Prior / Rejection sampling**: sample from the prior, discard samples inconsistent with evidence. Unbiased but wasteful when evidence is unlikely.
- **Likelihood weighting**: fix evidence variables, weight each sample by the likelihood of the evidence. No rejections.
- **Gibbs sampling (MCMC)**: start from a state consistent with evidence, repeatedly resample each non-evidence variable from `P(Xi | MarkovBlanket(Xi))`. Explain burn-in, thinning, and the Markov blanket.
- *(Optional stretch)* **Metropolis-Hastings** and **Mean-field variational inference** (short notebook only).

### 2.6 Learning from data
- **MLE**: `θ = count(x, pa) / count(pa)` — fails on zero counts.
- **Bayesian estimation with Dirichlet prior (Laplace/BDeu smoothing)**: `θ = (count + α) / (total + α·k)`.
- Show the effect of dataset size and prior strength on the learned CPTs.

### 2.7 Evaluation
Accuracy, F1, log-loss, Brier score, calibration curve (Bayesian predictions should be well calibrated — show it). Compare exact vs. approximate with **KL divergence** and **mean absolute error** as a function of number of samples.

---

## 3. Data

### 3.1 Primary dataset
**UCI "Student Performance" dataset** (Cortez & Silva, 2008) — `student-mat.csv` and `student-por.csv`, semicolon separated.
Download from: https://archive.ics.uci.edu/dataset/320/student+performance

Place under `data/raw/`. Do **not** commit large data; add a `scripts/download_data.py` that fetches and unzips it.

### 3.2 Synthetic data generator (required)
Implement `src/data/synthetic.py` that **samples from a known ground-truth BN** with hand-written CPTs. This lets us:
- verify that parameter learning recovers the true CPTs,
- verify that approximate inference converges to the exact answer,
- run all tests with no network access.

### 3.3 Discretization
Bayesian networks here use **discrete variables with 2–4 states**. Implement `src/data/preprocess.py`:

| Raw column | Variable | States |
|---|---|---|
| `studytime` (1–4) | `StudyTime` | Low, Medium, High |
| `absences` | `Attendance` | Poor (>10 abs), Average (4–10), Good (<4) |
| `failures` | `PastFailures` | None, Some |
| `G1` (0–20) | `PriorGrade` | Low (<10), Mid (10–14), High (≥15) |
| `Medu`, `Fedu` | `ParentEdu` | Low, High (max of both) |
| `famsup`, `schoolsup` | `Support` | None, Some |
| `higher` | `Motivation` | Low, High |
| `Dalc`, `goout` | `Distraction` | Low, High |
| `internet` | `Internet` | No, Yes |
| `G3` (0–20) | `FinalGrade` | Fail (<10), Pass (10–14), Excellent (≥15) |

Store the mapping in `configs/discretization.yaml`. Document each choice and its bias risks.
Split data: 70% train / 30% test, stratified on `FinalGrade`, fixed seed.

---

## 4. Network Structure

Define the structure in `configs/network.yaml` (edges list) so it is editable without code changes.

**Default hand-designed structure:**

```
ParentEdu ──► Support ──► StudyTime ──► FinalGrade
    │                        ▲              ▲
    └──────► Internet        │              │
                             │              │
Motivation ──────────────────┤              │
Motivation ──► Attendance ───┼──────────────┤
Distraction ─► StudyTime     │              │
PastFailures ─► PriorGrade ──┴──────────────┘
```

Edge list:
```
ParentEdu → Support
ParentEdu → Internet
Support → StudyTime
Motivation → StudyTime
Motivation → Attendance
Distraction → StudyTime
Distraction → Attendance
PastFailures → PriorGrade
StudyTime → FinalGrade
Attendance → FinalGrade
PriorGrade → FinalGrade
```

Rules:
- Must be a DAG (assert in code).
- Keep max in-degree ≤ 3 so CPTs stay small and readable.
- Also support **structure learning** (Phase 8, optional): Hill Climbing with BIC score via `pgmpy`, then compare with the hand-designed graph.

---

## 5. Tech Stack

- Python 3.10+
- `numpy`, `pandas`, `scipy`
- `pgmpy` — reference implementation / cross-checking only
- `networkx` — graph handling and drawing
- `matplotlib` (no seaborn required)
- `scikit-learn` — baselines and metrics
- `streamlit` — interactive demo app
- `pytest` — tests
- `jupyter` — teaching notebooks
- `pyyaml`, `tqdm`

Pin versions in `requirements.txt`. Use a `pyproject.toml` with `ruff` for linting.

---

## 6. Repository Structure

```
student-performance-bayes/
├── AGENTS.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── network.yaml
│   └── discretization.yaml
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── download_data.py
│   └── run_experiments.py
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   └── synthetic.py
│   ├── bn/
│   │   ├── network.py
│   │   ├── factor.py
│   │   ├── learning.py
│   │   └── structure.py
│   ├── inference/
│   │   ├── exact.py
│   │   ├── sampling.py
│   │   ├── gibbs.py
│   │   └── pgmpy_ref.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── compare.py
│   ├── baselines.py
│   └── viz.py
├── notebooks/
│   ├── 01_bayes_theorem.ipynb
│   ├── 02_bayesian_network.ipynb
│   ├── 03_exact_inference.ipynb
│   ├── 04_approximate_inference.ipynb
│   ├── 05_learning_parameters.ipynb
│   └── 06_prediction_and_evaluation.ipynb
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_factor.py
│   ├── test_exact.py
│   ├── test_sampling.py
│   ├── test_gibbs.py
│   ├── test_learning.py
│   └── test_against_pgmpy.py
└── reports/
    └── figures/
```

---

## 7. Build Phases

Complete each phase, run its tests, and commit before moving on.

### Phase 0 — Setup
- Create the structure, `requirements.txt`, `pyproject.toml`, `.gitignore`.
- Write `scripts/download_data.py`.

### Phase 1 — Bayes' theorem warm-up
- `notebooks/01_bayes_theorem.ipynb`: medical-test style example, then a student example (`P(Fail | LowAttendance)`).
- Include a posterior-vs-prior interactive plot (sliders optional).

### Phase 2 — Core data structures
- `Factor` class using NumPy arrays with named axes: `product`, `marginalize`, `reduce(evidence)`, `normalize`.
- `BayesNet` class: add nodes with state lists, parents, CPT arrays; validate CPT columns sum to 1; compute joint probability of a full assignment; topological order; Markov blanket.
- Tests: factor ops on tiny hand-computed examples.

### Phase 3 — Synthetic data + ground-truth network
- Hand-write CPTs for the default structure in `synthetic.py`.
- Implement forward (ancestral) sampling to generate N rows.
- Test: empirical frequencies of 100k samples ≈ true marginals (tolerance 0.01).

### Phase 4 — Exact inference
- `enumeration_ask(bn, query, evidence)`
- `variable_elimination(bn, query, evidence, order=None)`; support a heuristic order (min-neighbors / min-fill) and a user-specified order.
- Log the size of the largest intermediate factor to show the effect of ordering.
- Tests: VE == enumeration == `pgmpy.VariableElimination` to 1e-9 on at least 10 query/evidence combinations, including the explaining-away case.

### Phase 5 — Approximate inference
- `rejection_sampling(bn, query, evidence, n)` — also report the **acceptance rate**.
- `likelihood_weighting(bn, query, evidence, n)` — also report **effective sample size**: `ESS = (Σw)² / Σw²`.
- `gibbs_sampling(bn, query, evidence, n, burn_in, thin, seed)` — resample from the Markov blanket conditional.
- All accept a `seed` for reproducibility and return both the estimated distribution and diagnostics.
- Tests: with n = 200k, estimates are within 0.02 of exact answers.

### Phase 6 — Learning
- `fit_mle(structure, data)` and `fit_bayesian(structure, data, alpha)`.
- Experiment: learn from synthetic data at N = 50, 200, 1000, 10000 and plot the error between learned and true CPTs for MLE vs. Dirichlet α = 1.
- Handle unseen parent configurations (uniform fallback) and document it.

### Phase 7 — Prediction on real data & evaluation
- Fit the BN on the preprocessed UCI data.
- Predict `FinalGrade` for each test student via exact inference given all other observed variables; also with partial evidence (to show the BN handles missing data naturally).
- Compare against Naive Bayes and Logistic Regression: accuracy, macro-F1, log-loss, Brier score, calibration plot.
- Run the same prediction with likelihood weighting / Gibbs and report the accuracy gap vs. exact.

### Phase 8 — Experiments: exact vs. approximate
`scripts/run_experiments.py` must produce, in `reports/figures/`:
1. KL divergence vs. number of samples (log-log) for rejection, LW, Gibbs.
2. Runtime vs. number of samples.
3. Rejection acceptance rate vs. evidence rarity.
4. Gibbs trace plots and autocorrelation; effect of burn-in.
5. Variable elimination cost vs. elimination order.
6. *(Optional)* Structure learning (HillClimb + BIC) vs. hand-designed graph.

### Phase 9 — Streamlit app
`app/streamlit_app.py`:
- Sidebar: choose evidence for each variable (or "unobserved").
- Show the network graph with evidence nodes highlighted.
- Show the posterior of `FinalGrade` as a bar chart.
- Toggle inference method (Exact / Rejection / Likelihood Weighting / Gibbs) and sample size; show the exact answer alongside so the user can see approximation error.
- A "What-if" panel: change one variable (e.g. StudyTime Low → High) and show the change in P(Pass).
- Short "Explain this" expander on each method.

### Phase 10 — Docs
- `README.md` with: motivation, concept summary, architecture diagram, how to run, results table, figures, limitations.
- Each notebook must start with learning objectives and end with a recap.

---

## 8. Coding Standards

- Type hints and docstrings on every public function. Docstrings state the **math** being implemented.
- Small, pure functions; no global state. Randomness only through `np.random.default_rng(seed)`.
- Work in **log-space** where products of many probabilities occur (likelihood weighting, Gibbs) to avoid underflow.
- Never hard-code variable names inside algorithms; they must work on any `BayesNet`.
- Add readable `__repr__`/`print_cpt()` for CPTs so students can inspect them.
- Keep functions < 50 lines where reasonable; comment the *why*, not the *what*.
- Lint with `ruff`; all tests must pass with `pytest -q`.

---

## 9. Acceptance Criteria

The project is done when all of these hold:

- [ ] `pytest` passes, including cross-checks against `pgmpy`.
- [ ] From-scratch VE matches `pgmpy` to 1e-9.
- [ ] With ≥100k samples, all three approximate methods are within 0.02 of exact on every test query.
- [ ] Learned CPTs converge to ground truth as N grows (plot included).
- [ ] BN achieves accuracy within a few points of the best baseline on UCI data **and** has better log-loss/calibration than Naive Bayes (report honestly if not).
- [ ] All six notebooks run top-to-bottom with no errors in a fresh environment.
- [ ] Streamlit app launches with `streamlit run app/streamlit_app.py`.
- [ ] README explains every concept in Section 2 with at least one figure or worked example.

---

## 10. Things to Be Careful About (explain these in the README too)

- **Discretization loses information**; show a sensitivity check with different bin thresholds.
- **Data leakage**: `G1` and `G2` strongly predict `G3`. Report results both with and without `G2`; the default network excludes `G2` so the model is useful for *early* prediction.
- **Causality caveat**: edges in the hand-built graph are plausible assumptions, not proven causal claims. Do not present the output as causal.
- **Fairness**: variables like parental education may encode socio-economic bias. Add a short discussion and avoid using the model for real high-stakes decisions.
- **Small data**: the UCI math set has ~395 rows, so CPTs with many parents are poorly estimated; this is why smoothing and low in-degree matter.
- **Gibbs pitfalls**: deterministic or near-deterministic CPTs can break ergodicity; document and test.

---

## 11. Agent Working Rules

1. Work phase by phase; after each phase run tests and summarize what was done in a short commit message.
2. If a design choice is ambiguous, choose the simplest option that keeps the teaching value, and record the decision in `docs/DECISIONS.md`.
3. Never skip the from-scratch implementation by calling `pgmpy` directly — `pgmpy` is for validation only.
4. Do not fabricate results. All numbers and plots in the README must come from actually running the scripts.
5. Keep notebooks short, narrative, and runnable; move heavy code into `src/`.
6. Ask the user before adding any dependency not listed in Section 5.

---

## 12. Suggested First Prompt to the Agent

> Read AGENTS.md. Start with Phase 0 and Phase 1, then stop and show me the repo tree and the first notebook. Wait for my confirmation before starting Phase 2.
