# SIGNOVA Phase 11 — Annotation Quality Policy & Verification Standards

---

## 1. Core Principle: Structural Validity $\neq$ Linguistic Verification

A syntactically valid JSON file does NOT grant an annotation `VERIFIED` status. Scientific integrity requires that ground-truth status is awarded only when backed by genuine human reviewer evidence.

| Quality Grade | Criteria | Training Eligibility |
| :--- | :--- | :---: |
| **`UNVERIFIED`** | Raw draft or unvalidated annotation. | **`False`** |
| **`WEAK`** | Incomplete sequence, malformed tokens, or invalid temporal boundaries. | **`False`** |
| **`PARTIAL`** | Structurally valid and uppercase-compliant, but unreviewed. | **`False`** |
| **`VERIFIED`** | Structurally valid AND verified by an independent human reviewer. | **`True`** (if split & provenance valid) |
| **`LINGUIST_REVIEWED`** | Certified by a Deaf ISL specialist or linguist. | **`True`** |

---

## 2. Deterministic Training Eligibility Rule

$$\text{Training Eligible} \iff \begin{cases}
\text{glosses is non-empty} \\
\text{all tokens pass standard lexical regex} \\
\text{review\_status} = \text{VERIFIED} \\
\text{reviewer\_id is present} \\
\text{quality\_grade} \in \{\text{VERIFIED}, \text{LINGUIST\_REVIEWED}\} \\
\text{dataset\_split} \in \{\text{train}, \text{val}, \text{test}\} \\
\text{provenance is verified}
\end{cases}$$

No annotation may be promoted to training readiness by manually setting flags without passing these validation checks.
