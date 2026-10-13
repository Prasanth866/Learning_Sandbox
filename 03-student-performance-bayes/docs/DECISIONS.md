# Architecture and Design Decisions (DECISIONS.md)

This log records major design and pedagogical choices made throughout the implementation of the Student Performance Bayesian Network project.

---

## Decision 1: Python Virtual Environment & Dependency Management
- **Context**: The project requires a reproducible Python 3.10+ setup with core scientific libraries (`numpy`, `pandas`, `scipy`, `pgmpy`, `networkx`, `scikit-learn`, `streamlit`, `jupyter`).
- **Decision**: Use `uv` with Python 3.12 for fast, reproducible dependency resolution, maintaining `pyproject.toml` and pinned `requirements.txt`.
- **Rationale**: Python 3.12 has full wheel support across arm64 macOS for all specified libraries, avoiding compilation issues while staying modern.

## Decision 2: Discretization Strategy
- **Context**: Bayesian networks require discrete conditional probability tables (CPTs) with manageable dimension (keep state cardinalities between 2 and 3).
- **Decision**: Follow Section 3.3 discretization table precisely:
  - `StudyTime`: Low (<2 hrs/week, mapped from 1), Medium (2-5 hrs/week, mapped from 2), High (>5 hrs/week, mapped from 3, 4).
  - `Attendance`: Poor (>10 absences), Average (4-10 absences), Good (<4 absences).
  - `PastFailures`: None (0), Some (>0).
  - `PriorGrade`: Low (<10), Mid (10-14), High (>=15).
  - `ParentEdu`: Low (max of Medu, Fedu < 3), High (max of Medu, Fedu >= 3).
  - `Support`: None (no school/fam sup), Some (either school or fam sup).
  - `Motivation`: Low (`higher == 'no'`), High (`higher == 'yes'`).
  - `Distraction`: Low (`Dalc + goout <= 4`), High (`Dalc + goout > 4`).
  - `Internet`: No, Yes.
  - `FinalGrade`: Fail (<10), Pass (10-14), Excellent (>=15).
- **Rationale**: Keeps node in-degree <= 3 so maximum CPT size per node is $3 \times 3 \times 3 \times 3 = 81$ entries, preventing extreme sparsity and exponential memory blowup on ~395 student instances.

## Decision 3: DAG Topology & Exclusion of G2
- **Context**: In the UCI dataset, G2 is intermediate second-period grade, which trivially predicts G3 (FinalGrade).
- **Decision**: Default network structure uses `PriorGrade` (derived from G1) and excludes G2 for early intervention forecasting, as mandated in Section 10 of AGENTS.md.
