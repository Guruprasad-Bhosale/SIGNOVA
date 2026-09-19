# Phase 24: Held-Out Evaluation & Error Analysis Protocol

## Overview

The Phase 24 evaluation protocol guarantees scientifically rigorous, non-leaking evaluation of genuine CTC sign recognition models on held-out test partitions.

## Test Partition Isolation Invariant

- **Zero Leakage**: $\text{Train} \cap \text{Test} = \emptyset$ and $\text{Validation} \cap \text{Test} = \emptyset$.
- **Signer Independence**: When a signer-independent split strategy is designated, $\text{Signer}_{\text{Train}} \cap \text{Signer}_{\text{Test}} = \emptyset$.
- **Refusal Semantics**: `--phase24-eval` explicitly refuses execution if no verified model checkpoint exists.

## Primary Scientific Metrics

| Metric | Full Name | Target Threshold | Description |
| :--- | :--- | :--- | :--- |
| **TER** | Token Error Rate | Baseline Benchmark | Levenshtein edit distance over gloss sequence tokens normalized by reference length. |
| **EM** | Exact Match Accuracy | Baseline Benchmark | Exact string-token match between predicted and target gloss sequence. |
| **Token F1** | Micro/Macro Gloss F1 | Baseline Benchmark | Precision/Recall balance over vocabulary occurrences. |
| **FER** | Frame Error Rate | Diagnostic | Frame-level classification accuracy (when frame alignments exist). |

## Error Typology Breakdown

1. **Insertion Errors ($I$)**: Over-segmentation or spurious token emissions during sign transitions.
2. **Deletion Errors ($D$)**: Missed fast gestures, incomplete sign executions, or under-segmentation.
3. **Substitution Errors ($S$)**: Morphologically similar handshapes or movement confusions.
4. **Length Discrepancies**: Temporal compression or dilation discrepancies under variable signing speed.
5. **Confidence Gating & Silence/Inactivity**: Correct suppression and abstention during non-signing baseline resting poses.
