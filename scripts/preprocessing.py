"""
Converts raw (audio, text) examples into Whisper model inputs:
  - input_features: log-mel spectrogram from WhisperFeatureExtractor
    (with SpecAugment applied during training)
  - labels: tokenized, normalized transcript

Also defines the data collator that pads batches and masks padded label
tokens with -100 so they don't contribute to the loss.
"""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer

import config
from augmentation import maybe_speed_perturb, spec_augment
from text_normalize import normalize_text


def build_prepare_fn(
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    is_training: bool,
):
    """Returns a function suitable for dataset.map(..., batched=False)."""

    def prepare(example: dict) -> dict:
        audio = example["audio"]
        array = audio["array"]
        sr = audio["sampling_rate"]
        lang = example.get("lang", "")

        if is_training:
            array = maybe_speed_perturb(array, sr, lang)

        features = feature_extractor(
            array, sampling_rate=sr, return_tensors="np"
        ).input_features[0]

        if is_training:
            features = spec_augment(features, lang)

        example["input_features"] = features

        text = normalize_text(example["text"])
        example["labels"] = tokenizer(text).input_ids

        return example

    return prepare


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, batch: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in batch]
        batch_inputs = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in batch]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip the BOS token if the tokenizer already prepended it AND the
        # collator's padding logic added it again — avoids the classic
        # Whisper fine-tuning double-BOS bug.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch_inputs["labels"] = labels
        return batch_inputs
