"""
Run this AFTER training completes (separate Kaggle push/run — inference is
cheap enough to not need to be bundled into the same session as training).

Loads your fine-tuned model(s) from Hugging Face, transcribes the Zindi test
set, optionally applies KenLM shallow fusion (Strategy 3), applies text
normalization (Strategy 4), and writes a submission CSV.

CONFIRMED FROM REAL ZINDI FILES (Test.csv / Test_Phase2.csv, inspected
directly):
  - Phase 1 Test.csv:  IDs are language-prefixed (lug_96114, lin_24783,
    sna_55007) — language is known from the ID itself, no detection needed.
  - Phase 2 Test_Phase2.csv: IDs are OPAQUE (ID_QNYPTX, ID_CLCVQW, ...),
    columns are "ID,Target" — no language prefix at all. Which of your two
    fine-tuned checkpoints (sna_lug_combined vs lin_solo) applies to a given
    clip is NOT knowable from the ID and must be predicted from the audio.
    This is why LANGID_ENABLED exists in config.py — it is not optional
    for Phase 2, only skippable for Phase 1.

KenLM setup is the one piece that needs a one-time manual step: training
the n-gram model itself. See train_kenlm() below — run it once on your
training transcripts, upload the resulting .arpa/.bin file as a Kaggle
dataset, and point KENLM_MODEL_PATH at it.
"""

import logging
import os
import pandas as pd
import torch
from datasets import load_dataset, Audio

from transformers import WhisperForConditionalGeneration, WhisperProcessor

import config
from text_normalize import normalize_text
from hf_push import get_repo_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

KENLM_MODEL_PATH = os.environ.get("WAXAL_KENLM_PATH", None)  # set once you've built it


def train_kenlm(transcripts: list[str], output_path: str, n: int = 5):
    """
    One-time step: extract training transcripts -> plain text file -> train
    a KenLM n-gram model. Requires the `kenlm` toolkit built (via
    `pip install https://github.com/kpu/kenlm/archive/master.zip` plus the
    lmplz binary, or the conda-forge kenlm package which ships lmplz).

    This is NOT run automatically as part of run_pipeline.py — do it once,
    interactively, after you've decided your final training text corpus.
    """
    txt_path = output_path.replace(".arpa", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in transcripts:
            f.write(normalize_text(line) + "\n")

    cmd = f"lmplz -o {n} < {txt_path} > {output_path}"
    logger.info(f"Run this manually in a shell cell: {cmd}")
    return cmd


def load_finetuned_model(hf_repo_id: str):
    logger.info(f"Loading fine-tuned model from {hf_repo_id}")
    processor = WhisperProcessor.from_pretrained(hf_repo_id)
    model = WhisperForConditionalGeneration.from_pretrained(hf_repo_id)
    model.eval()
    return model, processor


def load_all_checkpoints(hf_username: str) -> dict:
    """
    Loads both fine-tuned checkpoints (Shona+Luganda combined, Lingala solo)
    keyed by the run name, so inference can route each clip to the right one.
    Both are needed for Phase 2 regardless of LANGID_METHOD, since even the
    "all_checkpoints" voting method needs every checkpoint loaded.
    """
    run_names = set(config.WHISPER_LANGID_TO_RUN.values()) | {config.LANGID_FALLBACK_RUN}
    checkpoints = {}
    for run_name in run_names:
        repo_id = f"{hf_username}/{config.HF_REPO_NAME_TEMPLATE.format(run_name=run_name)}"
        model, processor = load_finetuned_model(repo_id)
        checkpoints[run_name] = (model, processor)
    return checkpoints


# A generic (non-fine-tuned) Whisper processor purely for its language
# detection — cheap, one forward pass, avoids needing a separate trained
# classifier. Loaded lazily since not every run needs it (Phase 1 doesn't).
_langid_model = None
_langid_processor = None


def _get_langid_model():
    global _langid_model, _langid_processor
    if _langid_model is None:
        logger.info(f"Loading {config.BASE_MODEL_ID} for language detection only")
        _langid_processor = WhisperProcessor.from_pretrained(config.BASE_MODEL_ID)
        _langid_model = WhisperForConditionalGeneration.from_pretrained(config.BASE_MODEL_ID)
        _langid_model.eval()
    return _langid_model, _langid_processor


def detect_language(audio_array, sampling_rate: int) -> str:
    """
    Returns the run_name to use for this clip ("sna_lug_combined" or
    "lin_solo"), based on Whisper's own built-in language detection.

    Whisper's generate() can be asked to only run the language-ID step
    (no full transcription) by passing language=None and reading back the
    detected token from the first generated id — this is the standard
    lightweight way to get Whisper's language guess without a full decode.
    """
    model, processor = _get_langid_model()
    input_features = processor.feature_extractor(
        audio_array, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features

    with torch.no_grad():
        # detect_language() is a small helper on recent transformers
        # WhisperForConditionalGeneration purpose-built for exactly this.
        lang_token_ids = model.detect_language(input_features)
        detected = processor.tokenizer.decode(lang_token_ids[0].unsqueeze(0))
        # decode gives something like "<|ln|>" — strip the special-token wrapper
        iso = detected.strip("<|>")

    run_name = config.WHISPER_LANGID_TO_RUN.get(iso, config.LANGID_FALLBACK_RUN)
    if iso not in config.WHISPER_LANGID_TO_RUN:
        logger.warning(
            f"Detected language '{iso}' not in WHISPER_LANGID_TO_RUN "
            f"(expected ln/sn/lg) — falling back to {config.LANGID_FALLBACK_RUN}. "
            f"This may mean Phase 2 includes a language outside your training "
            f"set — worth checking the Zindi discussion board."
        )
    return run_name


_ID_LANG_PREFIX_TO_RUN = {
    "lug": "sna_lug_combined",
    "sna": "sna_lug_combined",
    "lin": "lin_solo",
}


def _run_name_from_id(example_id: str) -> str | None:
    """
    Phase 1 shortcut: Test.csv IDs are confirmed language-prefixed
    (lug_96114, lin_24783, sna_55007) — no detection needed, just read the
    prefix. Returns None for Phase 2's opaque IDs (ID_XXXXXX), signaling
    the caller to fall back to audio-based detection.
    """
    prefix = example_id.split("_")[0]
    return _ID_LANG_PREFIX_TO_RUN.get(prefix)


def transcribe_dataset(checkpoints: dict, test_dataset, id_column: str = "ID", use_kenlm: bool = False):
    """
    Transcribes every clip, routing each one to the correct fine-tuned
    checkpoint. Tries the cheap ID-prefix shortcut first (Phase 1); falls
    back to audio-based language detection when the ID is opaque (Phase 2,
    confirmed from the real Test_Phase2.csv format).

    checkpoints: dict of run_name -> (model, processor), from load_all_checkpoints()

    NOTE on KenLM shallow fusion: transformers' built-in generate() does not
    natively support n-gram shallow fusion out of the box for Whisper the
    way it does for Wav2Vec2/CTC models — shallow fusion here typically
    means re-scoring generate()'s beam search candidates with the KenLM
    score rather than fusing at every decoding step. If you want proper
    per-step fusion, pyctcdecode's BeamSearchDecoderCTC is the standard
    tool but it's built for CTC output, not Whisper's seq2seq output — so
    for Whisper, plan on an n-best rescoring approach instead:
        1. generate(num_beams=5, num_return_sequences=5) to get 5 candidates
        2. score each candidate's text with the KenLM model
        3. pick argmax(acoustic_score + lm_weight * kenlm_score)
    This function currently returns the top beam only (no rescoring) —
    treat KenLM rescoring as a follow-up step once base inference works.
    """
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    predictions = []
    langid_used_count = 0

    for example in test_dataset:
        audio = example["audio"]
        example_id = str(example[id_column])

        # Always try the free ID-prefix shortcut first (works for Phase 1's
        # lug_/lin_/sna_ IDs). Only fall back to audio-based detection when
        # the ID is opaque (Phase 2's ID_XXXXXX format) and LANGID_ENABLED.
        run_name = _run_name_from_id(example_id)
        if run_name is None:
            if not config.LANGID_ENABLED:
                raise RuntimeError(
                    f"ID '{example_id}' has no recognizable language prefix "
                    f"and LANGID_ENABLED=False in config.py — enable it to "
                    f"handle opaque IDs (e.g. Phase 2)."
                )
            run_name = detect_language(audio["array"], audio["sampling_rate"])
            langid_used_count += 1

        model, processor = checkpoints[run_name]

        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features

        generated_ids = model.generate(input_features, num_beams=5, max_length=225)
        text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(normalize_text(text))

    if langid_used_count:
        logger.info(
            f"Audio-based language detection used for {langid_used_count}/"
            f"{len(test_dataset)} clips (opaque IDs, i.e. Phase 2)."
        )

    return predictions


def load_test_audio(test_csv_path: str, audio_source: str) -> "Dataset":
    """
    Loads the test set and joins audio to each ID, auto-detecting which of
    the common Zindi delivery formats `audio_source` actually is:

      (a) A directory of individual audio files named by ID
          (e.g. audio_source/ID_QNYPTX.wav) — most common Zindi ASR format.
      (b) A single archive (.zip/.tar/.tar.gz) containing files named by ID
          — extracted once into a sibling "<audio_source>_extracted" dir,
          then handled as case (a).
      (c) A HuggingFace dataset repo id (contains "/", no local path exists)
          — loaded via datasets.load_dataset(), matched to test_csv_path's
          IDs by an "id"/"ID" column if present.

    Which of these applies is genuinely unknown until you check Zindi's
    Data page — this function does NOT guess blindly; it inspects
    audio_source and raises a clear, specific error if none of the three
    patterns match, rather than silently producing wrong results.
    """
    from datasets import Dataset
    import glob
    import tarfile
    import zipfile

    test_df = pd.read_csv(test_csv_path)
    ids = test_df["ID"].astype(str).tolist()

    # --- Case (c): looks like a HF repo id, not a local path ---
    if "/" in audio_source and not os.path.exists(audio_source):
        logger.info(f"Treating audio_source='{audio_source}' as a Hugging Face dataset repo")
        hf_ds = load_dataset(audio_source, split="test")
        id_col = "id" if "id" in hf_ds.column_names else ("ID" if "ID" in hf_ds.column_names else None)
        if id_col is None:
            raise RuntimeError(
                f"HF dataset '{audio_source}' has no 'id'/'ID' column to join "
                f"against {test_csv_path}'s IDs — inspect its columns "
                f"({hf_ds.column_names}) and adjust this join manually."
            )
        hf_by_id = {str(row[id_col]): row for row in hf_ds}
        missing = [i for i in ids if i not in hf_by_id]
        if missing:
            raise RuntimeError(
                f"{len(missing)}/{len(ids)} test IDs not found in '{audio_source}' "
                f"(e.g. {missing[:3]}) — this may be the wrong dataset/split."
            )
        rows = [{"ID": i, "audio": hf_by_id[i]["audio"]} for i in ids]
        return Dataset.from_list(rows)

    # --- Case (b): a single archive file — extract, then fall through to (a) ---
    if os.path.isfile(audio_source) and (
        audio_source.endswith(".zip") or ".tar" in audio_source
    ):
        extract_dir = audio_source.rsplit(".", 1)[0] + "_extracted"
        if not os.path.isdir(extract_dir):
            logger.info(f"Extracting archive {audio_source} -> {extract_dir}")
            os.makedirs(extract_dir, exist_ok=True)
            if audio_source.endswith(".zip"):
                with zipfile.ZipFile(audio_source) as zf:
                    zf.extractall(extract_dir)
            else:
                with tarfile.open(audio_source) as tf:
                    tf.extractall(extract_dir)
        audio_source = extract_dir  # fall through to case (a) below

    # --- Case (a): a directory of files named by ID ---
    if os.path.isdir(audio_source):
        logger.info(f"Treating audio_source='{audio_source}' as a directory of per-ID audio files")
        # Match by filename stem so we don't need to guess the extension
        # (.wav / .mp3 / .flac all seen across Zindi ASR competitions).
        available = {}
        for path in glob.glob(os.path.join(audio_source, "**", "*"), recursive=True):
            if os.path.isfile(path):
                stem = os.path.splitext(os.path.basename(path))[0]
                available[stem] = path

        missing = [i for i in ids if i not in available]
        if missing:
            raise RuntimeError(
                f"{len(missing)}/{len(ids)} test IDs have no matching audio "
                f"file under '{audio_source}' (e.g. missing: {missing[:3]}). "
                f"Either the ID-to-filename convention differs from a plain "
                f"stem match, or this isn't the right directory — check a "
                f"few filenames manually with `ls {audio_source} | head`."
            )

        rows = [{"ID": i, "audio": available[i]} for i in ids]
        ds = Dataset.from_list(rows)
        return ds.cast_column("audio", Audio(sampling_rate=16000))

    raise RuntimeError(
        f"Could not determine the audio delivery format for "
        f"audio_source='{audio_source}'. It's not an existing directory, "
        f"a .zip/.tar archive, or an HF-repo-shaped string. Check Zindi's "
        f"Data page for the actual audio download and pass its real local "
        f"path (after downloading) or HF repo id here."
    )


def build_submission(test_csv_path: str, predictions: list[str], output_path: str):
    """
    Column names confirmed directly from Test_Phase2.csv: "ID,Target".
    (Phase 1's Test.csv only has "ID" — same target column name applies
    per the SampleSubmission format shown alongside it.)
    """
    test_df = pd.read_csv(test_csv_path)
    if len(predictions) != len(test_df):
        raise ValueError(
            f"Prediction count ({len(predictions)}) doesn't match test set "
            f"size ({len(test_df)}) — check you ran inference on the full, "
            f"correctly-ordered test set."
        )
    submission = pd.DataFrame({"ID": test_df["ID"], "Target": predictions})
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission written to {output_path}")


def run_inference_pipeline(hf_username: str, test_csv_path: str, audio_source: str, output_path: str):
    """
    End-to-end: load both checkpoints, load+join test audio (auto-detecting
    the delivery format via load_test_audio — directory of files, a zip/tar
    archive, or a HF dataset repo), transcribe every clip (routing per-clip
    via ID prefix or audio language detection), write the submission CSV.

    audio_source: whatever you have after checking Zindi's Data page —
        a local directory of downloaded audio files, a path to a downloaded
        .zip/.tar archive, or a "namespace/dataset-name" HF repo id.
    """
    checkpoints = load_all_checkpoints(hf_username)
    test_dataset = load_test_audio(test_csv_path, audio_source)
    predictions = transcribe_dataset(checkpoints, test_dataset)
    build_submission(test_csv_path, predictions, output_path)


if __name__ == "__main__":
    print(
        "Example usage once you have both checkpoints trained:\n\n"
        "  from inference import run_inference_pipeline\n"
        "  run_inference_pipeline(\n"
        "      hf_username='your_hf_username',\n"
        "      test_csv_path='/kaggle/input/.../Test_Phase2.csv',\n"
        "      audio_source='/kaggle/input/.../test_audio',  # dir, .zip/.tar, or 'namespace/repo'\n"
        "      output_path='/kaggle/working/submission.csv',\n"
        "  )\n\n"
        "audio_source format is auto-detected (see load_test_audio docstring) —\n"
        "point it at whatever you actually find on Zindi's Data page.\n"
        "See docstrings in this file for the KenLM one-time setup step and\n"
        "the language-detection fallback used for Phase 2's opaque IDs."
    )
