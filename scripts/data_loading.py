"""
Loads WaxalNLP plus any enabled supplementary datasets (FLEURS, AfriVoice,
Common Voice) for the languages configured in config.py, standardizes them
to a common schema, and concatenates into one train / validation split.

Common schema after loading:
    {"audio": <Audio feature, 16kHz>, "text": str, "lang": str, "source": str}
"""

import logging
from datasets import (
    load_dataset,
    Audio,
    concatenate_datasets,
    Dataset,
    DatasetDict,
)

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_loading")

TARGET_SR = 16000


def _standardize(ds: Dataset, text_col: str, lang: str, source: str) -> Dataset:
    """Rename columns to the common schema and attach metadata columns."""
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")
    keep_cols = {"audio", "text"}
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    ds = ds.add_column("lang", [lang] * len(ds))
    ds = ds.add_column("source", [source] * len(ds))
    return ds


def load_waxal(lang_config: str) -> DatasetDict:
    """Load one language config from google/WaxalNLP (train/validation/test)."""
    logger.info(f"Loading WaxalNLP config '{lang_config}'")
    ds = load_dataset("google/WaxalNLP", lang_config)
    out = {}
    for split in ("train", "validation"):
        if split in ds:
            text_col = "text" if "text" in ds[split].column_names else "transcription"
            out[split] = _standardize(ds[split], text_col, lang_config, "waxal")
    return out


def load_fleurs(iso_code: str, lang_config: str) -> Dataset | None:
    if iso_code is None:
        return None
    try:
        logger.info(f"Loading FLEURS config '{iso_code}'")
        ds = load_dataset("google/fleurs", iso_code, split="train")
        return _standardize(ds, "transcription", lang_config, "fleurs")
    except Exception as e:
        logger.warning(f"FLEURS load failed for {iso_code}: {e}")
        return None


def load_afrivoice(iso_code: str, lang_config: str) -> Dataset | None:
    if iso_code is None:
        return None
    try:
        logger.info(f"Loading AfriVoice config '{iso_code}' (gated — requires prior HF agreement acceptance)")
        ds = load_dataset("DigitalUmuganda/AfriVoice", iso_code, split="train")
        text_col = "text" if "text" in ds.column_names else "transcription"
        return _standardize(ds, text_col, lang_config, "afrivoice")
    except Exception as e:
        logger.warning(
            f"AfriVoice load failed for {iso_code}: {e}. "
            "If this is a gating/permission error, accept the dataset's terms "
            "at https://huggingface.co/datasets/DigitalUmuganda/AfriVoice first."
        )
        return None


def load_common_voice(iso_code: str, lang_config: str) -> Dataset | None:
    if iso_code is None:
        return None
    try:
        logger.info(f"Loading Common Voice config '{iso_code}'")
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0", iso_code, split="train"
        )
        return _standardize(ds, "sentence", lang_config, "common_voice")
    except Exception as e:
        logger.warning(f"Common Voice load failed for {iso_code}: {e}")
        return None


def build_dataset_for_languages(languages: list[str]) -> DatasetDict:
    """
    Loads and mixes all enabled data sources for the given list of
    WaxalNLP language configs (e.g. ["sna_asr", "lug_asr"]).
    Returns a DatasetDict with "train" and "validation" splits.
    """
    train_parts, val_parts = [], []

    for lang_cfg in languages:
        iso = config.LANG_ISO.get(lang_cfg, {})

        waxal = load_waxal(lang_cfg)
        if "train" in waxal:
            train_parts.append(waxal["train"])
            logger.info(f"  {lang_cfg} / waxal / train: {len(waxal['train'])} examples")
        if "validation" in waxal:
            val_parts.append(waxal["validation"])
            logger.info(f"  {lang_cfg} / waxal / validation: {len(waxal['validation'])} examples")

        if config.USE_FLEURS:
            fleurs_ds = load_fleurs(iso.get("fleurs"), lang_cfg)
            if fleurs_ds is not None:
                train_parts.append(fleurs_ds)
                logger.info(f"  {lang_cfg} / fleurs / train: {len(fleurs_ds)} examples")

        if config.USE_AFRIVOICE:
            afri_ds = load_afrivoice(iso.get("afrivoice"), lang_cfg)
            if afri_ds is not None:
                train_parts.append(afri_ds)
                logger.info(f"  {lang_cfg} / afrivoice / train: {len(afri_ds)} examples")

        if config.USE_COMMON_VOICE:
            cv_ds = load_common_voice(iso.get("common_voice"), lang_cfg)
            if cv_ds is not None:
                train_parts.append(cv_ds)
                logger.info(f"  {lang_cfg} / common_voice / train: {len(cv_ds)} examples")

    if not train_parts:
        raise RuntimeError(
            "No training data loaded for any language — check dataset names, "
            "language configs, and HF authentication/gating."
        )

    train_ds = concatenate_datasets(train_parts).shuffle(seed=42)

    if val_parts:
        val_ds = concatenate_datasets(val_parts)
    else:
        # No validation split available anywhere — carve one out of train.
        logger.warning("No validation split found upstream; carving 5% out of train.")
        split = train_ds.train_test_split(test_size=0.05, seed=42)
        train_ds, val_ds = split["train"], split["test"]

    logger.info(f"Final mixed dataset — train: {len(train_ds)}, validation: {len(val_ds)}")

    return DatasetDict({"train": train_ds, "validation": val_ds})


if __name__ == "__main__":
    # Quick standalone check: run this file directly on Kaggle/Colab to see
    # real per-language, per-source example counts before committing to a
    # full training run.
    dd = build_dataset_for_languages(config.LANGUAGES)
    print(dd)
