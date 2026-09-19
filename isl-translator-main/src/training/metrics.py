"""
Evaluation Metrics for Sign Language Recognition.

Computes WER (Word Error Rate) and BLEU scores.
"""

import numpy as np
from typing import List, Optional, Tuple
import editdistance


def compute_wer(
    predictions: List[List[str]],
    references: List[List[str]],
) -> Tuple[float, dict]:
    """
    Compute Word Error Rate (WER) between prediction and reference sequences.
    
    WER = (Substitutions + Insertions + Deletions) / Reference Length
    
    Args:
        predictions: List of predicted gloss sequences
        references: List of reference gloss sequences
        
    Returns:
        wer: Overall WER
        details: Dictionary with per-sample and aggregate metrics
    """
    total_edits = 0
    total_ref_len = 0
    sample_wers = []
    
    for pred, ref in zip(predictions, references):
        # Compute Levenshtein distance
        edits = editdistance.eval(pred, ref)
        ref_len = len(ref)
        
        if ref_len > 0:
            sample_wer = edits / ref_len
        else:
            sample_wer = 0.0 if len(pred) == 0 else 1.0
        
        sample_wers.append(sample_wer)
        total_edits += edits
        total_ref_len += max(ref_len, 1)  # Avoid division by zero
    
    wer = total_edits / total_ref_len if total_ref_len > 0 else 0.0
    
    details = {
        "wer": wer,
        "total_edits": total_edits,
        "total_ref_length": total_ref_len,
        "sample_wers": sample_wers,
        "mean_sample_wer": np.mean(sample_wers) if sample_wers else 0.0,
    }
    
    return wer, details


def compute_cer(
    predictions: List[str],
    references: List[str],
) -> Tuple[float, dict]:
    """
    Compute Character Error Rate (CER) for translated text.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        cer: Overall CER
        details: Dictionary with metrics
    """
    total_edits = 0
    total_ref_len = 0
    
    for pred, ref in zip(predictions, references):
        edits = editdistance.eval(list(pred), list(ref))
        total_edits += edits
        total_ref_len += len(ref)
    
    cer = total_edits / total_ref_len if total_ref_len > 0 else 0.0
    
    return cer, {"cer": cer, "total_edits": total_edits, "total_ref_length": total_ref_len}


def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> Tuple[float, dict]:
    """
    Compute BLEU score for translation quality.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        max_n: Maximum n-gram order
        
    Returns:
        bleu: BLEU score (0-100)
        details: Dictionary with n-gram precisions
    """
    try:
        from sacrebleu.metrics import BLEU
        bleu_metric = BLEU()
        
        # sacrebleu expects list of references for each prediction
        refs = [[ref] for ref in references]
        
        result = bleu_metric.corpus_score(predictions, refs)
        
        return result.score, {
            "bleu": result.score,
            "precisions": result.precisions,
            "bp": result.bp,
            "sys_len": result.sys_len,
            "ref_len": result.ref_len,
        }
    except ImportError:
        # Fallback to simple BLEU implementation
        return _simple_bleu(predictions, references, max_n)


def _simple_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> Tuple[float, dict]:
    """Simple BLEU implementation without sacrebleu."""
    from collections import Counter
    import math
    
    precisions = []
    
    for n in range(1, max_n + 1):
        match_count = 0
        total_count = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            # Get n-grams
            pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
            
            pred_counter = Counter(pred_ngrams)
            ref_counter = Counter(ref_ngrams)
            
            # Count matches
            for ngram, count in pred_counter.items():
                match_count += min(count, ref_counter.get(ngram, 0))
            total_count += len(pred_ngrams)
        
        precision = match_count / total_count if total_count > 0 else 0
        precisions.append(precision)
    
    # Compute brevity penalty
    pred_len = sum(len(p.split()) for p in predictions)
    ref_len = sum(len(r.split()) for r in references)
    
    if pred_len == 0:
        bp = 0
    elif pred_len >= ref_len:
        bp = 1
    else:
        bp = math.exp(1 - ref_len / pred_len)
    
    # Compute BLEU
    if min(precisions) > 0:
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        bleu = 100 * bp * math.exp(log_avg)
    else:
        bleu = 0.0
    
    return bleu, {"bleu": bleu, "precisions": precisions, "bp": bp}


class MetricsTracker:
    """
    Track and aggregate training/validation metrics.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.predictions = []
        self.references = []
        self.translations = []
        self.text_references = []
    
    def update(
        self,
        loss: Optional[float] = None,
        predictions: Optional[List[List[str]]] = None,
        references: Optional[List[List[str]]] = None,
        translations: Optional[List[str]] = None,
        text_references: Optional[List[str]] = None,
    ):
        """Update metrics with batch results."""
        if loss is not None:
            self.losses.append(loss)
        if predictions is not None and references is not None:
            self.predictions.extend(predictions)
            self.references.extend(references)
        if translations is not None and text_references is not None:
            self.translations.extend(translations)
            self.text_references.extend(text_references)
    
    def compute(self) -> dict:
        """Compute aggregated metrics."""
        results = {}
        
        # Average loss
        if self.losses:
            results["loss"] = np.mean(self.losses)
        
        # WER
        if self.predictions and self.references:
            wer, wer_details = compute_wer(self.predictions, self.references)
            results["wer"] = wer
            results["wer_details"] = wer_details
        
        # BLEU (for translations)
        if self.translations and self.text_references:
            bleu, bleu_details = compute_bleu(self.translations, self.text_references)
            results["bleu"] = bleu
            results["bleu_details"] = bleu_details
        
        return results
    
    def summary(self) -> str:
        """Get summary string."""
        results = self.compute()
        parts = []
        
        if "loss" in results:
            parts.append(f"Loss: {results['loss']:.4f}")
        if "wer" in results:
            parts.append(f"WER: {results['wer']:.2%}")
        if "bleu" in results:
            parts.append(f"BLEU: {results['bleu']:.2f}")
        
        return " | ".join(parts)


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Test WER
    preds = [["HELLO", "WORLD"], ["HOW", "ARE", "YOU"]]
    refs = [["HELLO", "WORLD"], ["HOW", "IS", "YOU"]]
    
    wer, details = compute_wer(preds, refs)
    print(f"WER: {wer:.2%}")
    
    # Test BLEU
    pred_texts = ["hello world", "how are you"]
    ref_texts = ["hello world", "how is you"]
    
    bleu, details = compute_bleu(pred_texts, ref_texts)
    print(f"BLEU: {bleu:.2f}")
    
    # Test tracker
    tracker = MetricsTracker()
    tracker.update(loss=0.5, predictions=preds, references=refs)
    print(f"Summary: {tracker.summary()}")
    
    print("\nAll tests passed!")
