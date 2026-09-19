"""
Unit Tests for Phase 5 Data Ingestion, Label Normalization, and Vocabulary.
"""

from pathlib import Path
import pytest

from signova.data.adapters.sequential_adapter import GenericSequentialAdapter
from signova.data.vocabulary import SignVocabulary
from signova.preprocessing.label_normalizer import LabelNormalizer


def test_label_normalizer():
    normalizer = LabelNormalizer()

    # Test casing, whitespace, and punctuation normalization
    assert normalizer.normalize_token("  hello-world! ") == "HELLO-WORLD"
    assert normalizer.normalize_token("thank_you!!") == "THANK_YOU"

    # Test sequence normalization
    raw_toks, canon_toks = normalizer.normalize_sequence("HELLO, WORLD, ISL")
    assert raw_toks == ["HELLO", "WORLD", "ISL"]
    assert canon_toks == ["HELLO", "WORLD", "ISL"]


def test_sign_vocabulary_encoding_and_decoding(tmp_path: Path):
    vocab = SignVocabulary(blank_index=0, unk_token="<UNK>")
    assert len(vocab) == 2  # <BLANK>, <UNK>
    assert vocab.token_to_id["<BLANK>"] == 0
    assert vocab.token_to_id["<UNK>"] == 1

    # Add tokens
    id_hello = vocab.add_token("HELLO", split="train")
    id_world = vocab.add_token("WORLD", split="train")
    assert id_hello == 2
    assert id_world == 3

    # Encode sequence
    encoded = vocab.encode(["HELLO", "WORLD"])
    assert encoded == [2, 3]

    # Decode sequence
    decoded = vocab.decode([0, 2, 3], remove_blank=True)
    assert decoded == ["HELLO", "WORLD"]

    # Test OOV handling
    encoded_oov = vocab.encode(["HELLO", "UNKNOWN_SIGN"])
    assert encoded_oov == [2, 1]  # 1 is <UNK>

    # Test CSV save and reload
    csv_file = tmp_path / "test_vocab.csv"
    vocab.save_csv(csv_file)
    reloaded_vocab = SignVocabulary.load_csv(csv_file)
    assert len(reloaded_vocab) == len(vocab)
    assert reloaded_vocab.token_to_id["HELLO"] == 2


def test_generic_sequential_adapter_synthetic(tmp_path: Path):
    adapter = GenericSequentialAdapter(dataset_name="SYNTHETIC_TEST", is_synthetic=True)
    samples = [
        {
            "sample_id": "syn_01",
            "sequence_label": "HELLO, WORLD",
            "num_frames": 40,
            "feature_path": "fake.npz",
            "split": "train",
        }
    ]

    manifest_file = tmp_path / "canonical_manifest.csv"
    df = adapter.build_canonical_manifest(samples, output_csv=manifest_file)

    assert len(df) == 1
    assert bool(df.iloc[0]["is_synthetic"]) is True
    summary = adapter.summary()
    assert summary["integration_status"] == "SYNTHETIC_FIXTURE_VALIDATED"
    assert summary["is_synthetic"] is True
