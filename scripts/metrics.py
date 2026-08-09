"""
Word Error Rate computation, matching the challenge's evaluation metric.
Applies the same text normalization used in training/submission so the
eval-time WER you see is a realistic estimate of leaderboard WER.
"""

import evaluate
import numpy as np

from scripts.text_normalize import normalize_text

wer_metric = evaluate.load("wer")


def build_compute_metrics_fn(tokenizer):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalize_text(p) for p in pred_str]
        label_str = [normalize_text(l) for l in label_str]

        # Guard against empty references, which crash the WER computation.
        pairs = [(p, l) for p, l in zip(pred_str, label_str) if len(l) > 0]
        if not pairs:
            return {"wer": 1.0}
        preds, labels = zip(*pairs)

        wer = 100 * wer_metric.compute(predictions=list(preds), references=list(labels))
        return {"wer": wer}

    return compute_metrics
