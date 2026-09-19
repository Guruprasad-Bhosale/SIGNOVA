"""
Phase 13 CTC Training Pipeline Unit Tests.

NOTICE:
SYNTHETIC_FIXTURE_VALIDATION — NOT REAL ISL TRAINING.
Validates software training mechanics and ensures that under STATE_B
real training is defensively blocked with zero checkpoints produced.
"""

import numpy as np
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from signova.experiments.trainer import LandmarkSequenceDataset, RealCTCTrainer, collate_landmark_batch
from signova.qualification.constants import SUPERVISION_STATE_B
from signova.qualification.gate import GateEvaluationResult, RealCTCTrainingBlockedError
from signova.qualification.vocabulary import Phase12GlossVocabulary


def test_synthetic_fixture_training_mechanics():
    vocab = Phase12GlossVocabulary()
    vocab.token_to_id.update({"A": 2, "B": 3})
    vocab.id_to_token.update({2: "A", 3: "B"})

    # Synthetic fixture (SYNTHETIC_FIXTURE_VALIDATION — NOT REAL ISL TRAINING)
    samples = [
        (np.random.randn(20, 150).astype(np.float32), [2, 3], "synth_01"),
    ]
    dataset = LandmarkSequenceDataset(samples)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_landmark_batch)

    trainer = RealCTCTrainer(vocab=vocab, input_dim=150, hidden_dim=32, num_layers=1, seed=42)
    loss = trainer.train_epoch(loader)
    assert isinstance(loss, float)
    assert not np.isnan(loss)


def test_defensive_block_under_state_b_writes_no_artifacts(tmp_path):
    vocab = Phase12GlossVocabulary()
    trainer = RealCTCTrainer(vocab=vocab, input_dim=150, hidden_dim=32, num_layers=1)

    blocked_gate = GateEvaluationResult(
        supervision_state=SUPERVISION_STATE_B,
        real_ctc_status="BLOCKED",
        pilot_status="BLOCKED_HUMAN_RESOURCE",
        generalization_claims="NOT_READY",
        publication_grade_evaluation="NOT_READY",
        conditions_satisfied={},
        failed_conditions=["GENUINE_SEQUENTIAL_ANNOTATIONS_EXIST"],
        dataset_threshold_status={},
        total_annotations_found=0,
        training_eligible_count=0,
    )

    dataset = LandmarkSequenceDataset([])
    loader = DataLoader(dataset)

    out_dir = tmp_path / "exp"
    with pytest.raises(RealCTCTrainingBlockedError):
        trainer.run_experiment(loader, loader, loader, output_dir=out_dir, eval_result=blocked_gate)

    # Confirm no checkpoint was written
    assert not (out_dir / "best_model.pt").exists()
