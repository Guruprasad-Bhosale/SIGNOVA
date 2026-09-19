"""
Unit tests for Phase 7 Translation Data, Normalization, and Vocabulary.
"""

from pathlib import Path
import tempfile
import torch
from torch.utils.data import DataLoader

from signova.translation.dataset import (
    GlossTranslationDataset,
    TranslationPadCollate,
    TranslationSample,
)
from signova.translation.normalization import TranslationNormalizer
from signova.translation.vocabulary import TranslationVocabulary


def test_translation_normalizer():
    normalizer = TranslationNormalizer(lowercase_target=True, standardize_punctuation=True)
    
    # Gloss token normalization (uppercase, stripped)
    assert normalizer.normalize_gloss_token("  hello  ") == "HELLO"
    assert normalizer.normalize_gloss_sequence([" I ", "go", "college "]) == ["I", "GO", "COLLEGE"]

    # Target text normalization
    raw_text = '  “Hello, World!” — She said.  '
    norm_text = normalizer.normalize_english_text(raw_text)
    assert '"' in norm_text
    assert '-' in norm_text
    assert norm_text == '"hello, world!" - she said.'

    # Process pair
    processed = normalizer.process_pair(["I", "EAT"], "I eat food.")
    assert processed["normalized_source_tokens"] == ["I", "EAT"]
    assert processed["normalized_target_text"] == "i eat food."
    assert processed["raw_target_text"] == "I eat food."


def test_translation_vocabulary():
    vocab = TranslationVocabulary(name="test_vocab", is_synthetic=True)
    assert len(vocab) == 4
    assert vocab.pad_id == 0
    assert vocab.bos_id == 1
    assert vocab.eos_id == 2
    assert vocab.unk_id == 3

    sequences = [["I", "GO", "COLLEGE"], ["YOU", "GO", "HOME"]]
    vocab.build_from_sequences(sequences)

    assert "GO" in vocab
    assert "COLLEGE" in vocab
    assert "UNKNOWN" not in vocab

    # Encoding
    encoded = vocab.encode(["I", "GO", "UNKNOWN"], add_bos=True, add_eos=True)
    assert encoded[0] == vocab.bos_id
    assert encoded[-1] == vocab.eos_id
    assert encoded[-2] == vocab.unk_id

    # Decoding
    decoded = vocab.decode(encoded, skip_special=True)
    assert decoded == ["I", "GO", "<UNK>"]

    # Text decode
    text = vocab.decode_to_text(encoded, skip_special=True)
    assert text == "I GO <UNK>"

    # OOV rate
    oov_rate = vocab.calculate_oov_rate([["I", "NEW_SIGN", "UNKNOWN_2"]])
    assert round(oov_rate, 2) == 0.67

    # Save & Load roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "vocab.json"
        vocab.save(save_path)
        loaded_vocab = TranslationVocabulary.load(save_path)
        assert len(loaded_vocab) == len(vocab)
        assert loaded_vocab.encode(["I", "GO"]) == vocab.encode(["I", "GO"])


def test_gloss_translation_dataset_and_collate():
    samples = [
        TranslationSample(
            sample_id="SAMPLE-01",
            source_tokens=["I", "GO", "COLLEGE"],
            target_text="I go to college.",
            split="train",
        ),
        TranslationSample(
            sample_id="SAMPLE-02",
            source_tokens=["YOU", "HUNGRY"],
            target_text="Are you hungry?",
            split="train",
        ),
    ]

    src_vocab = TranslationVocabulary().build_from_sequences([s.source_tokens for s in samples])
    tgt_vocab = TranslationVocabulary().build_from_sequences([s.target_text.split() for s in samples])

    dataset = GlossTranslationDataset(
        samples,
        source_vocab=src_vocab,
        target_vocab=tgt_vocab,
    )
    assert len(dataset) == 2

    item = dataset[0]
    assert item["sample_id"] == "SAMPLE-01"
    assert "source_ids" in item
    assert "target_ids" in item

    collate_fn = TranslationPadCollate(source_pad_id=src_vocab.pad_id, target_pad_id=tgt_vocab.pad_id)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    batch = next(iter(loader))
    assert batch["source_ids"].shape == (2, 3)
    assert batch["source_lengths"].tolist() == [3, 2]
    assert batch["source_mask"].shape == (2, 3)
    assert batch["source_mask"][1, 2].item() is False  # Padding token masked out
