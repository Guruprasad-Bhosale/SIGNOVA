"""
Phase 12 Experiments, Baselines, Metrics, and Latency Package for SIGNOVA.
"""

from signova.experiments.baselines import (
    BaselineAMajoritySequence,
    BaselineBTemporalPoolingClassifier,
)
from signova.experiments.metrics import (
    SequenceMetricsResult,
    compute_levenshtein_breakdown,
    evaluate_sequence_predictions,
)
from signova.experiments.error_analysis import (
    SampleErrorRecord,
    perform_phase12_error_analysis,
)
from signova.experiments.latency import (
    LatencyProfileResult,
    profile_pipeline_latency,
)
from signova.experiments.trainer import (
    ContinuousBiGRUCTCModel,
    LandmarkSequenceDataset,
    RealCTCTrainer,
    collate_landmark_batch,
)

__all__ = [
    "BaselineAMajoritySequence",
    "BaselineBTemporalPoolingClassifier",
    "SequenceMetricsResult",
    "compute_levenshtein_breakdown",
    "evaluate_sequence_predictions",
    "SampleErrorRecord",
    "perform_phase12_error_analysis",
    "LatencyProfileResult",
    "profile_pipeline_latency",
    "ContinuousBiGRUCTCModel",
    "LandmarkSequenceDataset",
    "RealCTCTrainer",
    "collate_landmark_batch",
]
