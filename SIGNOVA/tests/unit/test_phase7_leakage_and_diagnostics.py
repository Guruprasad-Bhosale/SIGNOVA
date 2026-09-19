"""
Unit tests for Translation Leakage Detector and Translation Diagnostics.
"""

import torch
from torch.utils.data import DataLoader

from signova.translation.dataset import (
    GlossTranslationDataset,
    TranslationPadCollate,
    TranslationSample,
)
from signova.translation.diagnostics import TranslationDiagnostics
from signova.translation.leakage import TranslationLeakageDetector
from signova.translation.model import SyntheticGlossToEnglishSeq2Seq
from signova.translation.vocabulary import TranslationVocabulary


def test_translation_leakage_detector():
    clean_train = [
        TranslationSample(sample_id="TR-1", source_tokens=["I", "GO"], target_text="I go.", signer_id="S1"),
    ]
    clean_val = [
        TranslationSample(sample_id="VAL-1", source_tokens=["YOU", "GO"], target_text="You go.", signer_id="S2"),
    ]
    clean_test = [
        TranslationSample(sample_id="TS-1", source_tokens=["HE", "GO"], target_text="He goes.", signer_id="S3"),
    ]

    detector = TranslationLeakageDetector()
    report_clean = detector.check_splits(clean_train, clean_val, clean_test)
    assert report_clean.is_leakage_free is True
    assert report_clean.signer_overlap_detected is False

    # Contaminated split (exact pair overlap)
    dirty_val = [
        TranslationSample(sample_id="VAL-2", source_tokens=["I", "GO"], target_text="I go.", signer_id="S1"),
    ]
    report_dirty = detector.check_splits(clean_train, dirty_val, clean_test)
    assert report_dirty.is_leakage_free is False
    assert report_dirty.train_val_overlap_count == 1
    assert report_dirty.signer_overlap_detected is True


def test_translation_diagnostics():
    samples = [
        TranslationSample(sample_id="S1", source_tokens=["I", "EAT"], target_text="I eat.", split="val"),
        TranslationSample(sample_id="S2", source_tokens=["YOU", "DRINK"], target_text="You drink.", split="val"),
    ]

    src_vocab = TranslationVocabulary().build_from_sequences([s.source_tokens for s in samples])
    tgt_vocab = TranslationVocabulary().build_from_sequences([s.target_text.split() for s in samples])

    dataset = GlossTranslationDataset(samples, source_vocab=src_vocab, target_vocab=tgt_vocab)
    collate = TranslationPadCollate(src_vocab.pad_id, tgt_vocab.pad_id)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate)

    model = SyntheticGlossToEnglishSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_embed_dim=16,
        tgt_embed_dim=16,
        enc_hidden_dim=32,
        dec_hidden_dim=32,
    )

    diag_engine = TranslationDiagnostics(model, src_vocab, tgt_vocab, device="cpu")
    diag_report = diag_engine.evaluate_diagnostics(loader, is_synthetic=True)

    assert diag_report.total_samples == 2
    assert diag_report.teacher_forced_loss >= 0.0
    assert 0.0 <= diag_report.eos_termination_rate <= 100.0
    assert diag_report.avg_decoding_latency_ms >= 0.0
    assert diag_report.model_param_count > 0
