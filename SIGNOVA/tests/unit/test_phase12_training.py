"""
Phase 12 Trainer Pipeline Unit Tests.

NOTICE:
SYNTHETIC_FIXTURE_VALIDATION — NOT REAL CTC TRAINING.
This test validates software training mechanics on small synthetic fixtures.
Real CTC training strictly requires genuine STATE_A / STATE_A_DATA_LIMITED data.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from signova.experiments.trainer import LandmarkSequenceDataset, RealCTCTrainer, collate_landmark_batch
from signova.qualification.constants import SUPERVISION_STATE_A_DATA_LIMITED, SUPERVISION_STATE_B
from signova.qualification.gate import GateEvaluationResult, RealCTCTrainingBlockedError
from signova.qualification.vocabulary import Phase12GlossVocabulary


def test_synthetic_fixture_training_mechanics(tmp_path):
    # Setup test vocabulary
    vocab = Phase12GlossVocabulary()
    vocab.token_to_id.update({"NAMASTE": 2, "HELP": 3, "WATER": 4})
    vocab.id_to_token.update({2: "NAMASTE", 3: "HELP", 4: "WATER"})

    # Synthetic fixture data (SYNTHETIC_FIXTURE_VALIDATION — NOT REAL CTC TRAINING)
    samples = [
        (np.random.randn(20, 150).astype(np.float32), [2, 3], "synth_001"),
        (np.random.randn(25, 150).astype(np.float32), [4], "synth_002"),
    ]

    dataset = LandmarkSequenceDataset(samples)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_landmark_batch)

    trainer = RealCTCTrainer(
        vocab=vocab,
        input_dim=150,
        hidden_dim=32,
        num_layers=1,
        learning_rate=1e-2,
        seed=42,
    )

    # 1 Epoch of synthetic training mechanics
    loss = trainer.train_epoch(loader)
    assert not np.isnan(loss)
    assert isinstance(loss, float)

    # Evaluation
    eval_loss, refs, hyps, sids = trainer.evaluate(loader)
    assert not np.isnan(eval_loss)
    assert len(refs) == 2
    assert len(hyps) == 2


def test_trainer_blocks_when_passed_state_b_gate():
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

    with pytest.raises(RealCTCTrainingBlockedError):
        trainer.run_experiment(loader, loader, loader, eval_result=blocked_gate)
