"""
Phase 16 Training Pipeline & Provenance Unit Tests.

NOTICE:
SYNTHETIC_FIXTURE_VALIDATION — NOT REAL ISL TRAINING.
Validates:
- Training execution mechanics using synthetic fixtures.
- Defensive barrier under STATE_B preventing real model checkpoint creation.
- Checkpoint provenance manifest requirements (checkpoint_type = REAL_ISL_CTC_BASELINE vs SYNTHETIC).
- Invariant: synthetic_fixture_checkpoint != real_checkpoint.
"""

import numpy as np
import pytest
from torch.utils.data import DataLoader

from signova.experiments.trainer import LandmarkSequenceDataset, RealCTCTrainer, collate_landmark_batch
from signova.operations.checkpoint_provenance import (
    CHECKPOINT_TYPE_REAL,
    CHECKPOINT_TYPE_SYNTHETIC,
    create_real_checkpoint_provenance,
)
from signova.qualification.constants import SUPERVISION_STATE_B
from signova.qualification.gate import GateEvaluationResult, RealCTCTrainingBlockedError
from signova.qualification.vocabulary import Phase12GlossVocabulary


def test_synthetic_fixture_training_mechanics():
    """SYNTHETIC_FIXTURE_VALIDATION — NOT REAL ISL TRAINING."""
    vocab = Phase12GlossVocabulary()
    vocab.token_to_id.update({"NAMASTE": 2, "THANKYOU": 3})
    vocab.id_to_token.update({2: "NAMASTE", 3: "THANKYOU"})

    samples = [
        (np.random.randn(30, 150).astype(np.float32), [2, 3], "synth_p16_sample"),
    ]
    dataset = LandmarkSequenceDataset(samples)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_landmark_batch)

    trainer = RealCTCTrainer(vocab=vocab, input_dim=150, hidden_dim=32, num_layers=1, seed=42)
    loss = trainer.train_epoch(loader)
    assert isinstance(loss, float)
    assert not np.isnan(loss)


def test_defensive_barrier_blocks_training_under_state_b(tmp_path):
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

    out_dir = tmp_path / "blocked_experiment"
    with pytest.raises(RealCTCTrainingBlockedError):
        trainer.run_experiment(loader, loader, loader, output_dir=out_dir, eval_result=blocked_gate)

    assert not (out_dir / "best_model.pt").exists()


def test_real_checkpoint_provenance_structure():
    prov = create_real_checkpoint_provenance(
        dataset_version="p16_v1",
        annotation_version="gold_v1",
        vocabulary_version="vocab_v1",
        split_version="signer_indep_v1",
        feature_group="holistic_v1",
        seed=42,
        config_hash="sha256_mock_config",
    )
    d = prov.to_dict()
    assert d["checkpoint_type"] == CHECKPOINT_TYPE_REAL
    assert d["checkpoint_type"] != CHECKPOINT_TYPE_SYNTHETIC
    assert d["dataset_version"] == "p16_v1"
    assert d["training_seed"] == 42
    assert "created_at" in d
