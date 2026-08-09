"""
MASTER SCRIPT — runs the full WAXAL fine-tuning pipeline end-to-end:

    1. Authenticate to Hugging Face (checkpoints + final model destination)
    2. Load + mix datasets (WaxalNLP + enabled supplementary sources)
    3. Preprocess (normalize text, extract features, apply augmentation)
    4. Load Sunbird Whisper-51 in 4-bit NF4 + attach LoRA adapters
    5. Build LLRD optimizer param groups
    6. Train, with periodic checkpoint pushes to Hugging Face
    7. Push the final fine-tuned model to Hugging Face

This is what kernel-metadata.json points Kaggle at. Configure the run via
config.py (or the WAXAL_RUN_NAME / WAXAL_LANGUAGES env vars) BEFORE pushing —
e.g. one push for the Shona+Luganda combined run, a second push with
WAXAL_LANGUAGES=lin_asr for the dedicated Lingala run.

Run status/failure is NOT observable via the Kaggle API (see project notes) —
check the Kaggle notebook page directly to see whether this completed.
"""

import logging
import os
import sys

from scripts import config

# ---------------------------------------------------------------------------
# Kaggle's kernel runner invokes this file directly as `python run_pipeline.py`
# — there's no kernel-metadata.json field to make it launch under
# `accelerate` instead. To get real 2-GPU data-parallel training (see
# model_setup.py / training.py) rather than a single-GPU run that ignores
# the second T4, this script re-launches ITSELF via `accelerate launch`
# the first time it runs, then exits — the re-launched copy does the actual
# work. WAXAL_ACCELERATE_LAUNCHED guards against infinite re-launching.
# ---------------------------------------------------------------------------
if config.NUM_GPUS > 1 and os.environ.get("WAXAL_ACCELERATE_LAUNCHED") != "1":
    os.environ["WAXAL_ACCELERATE_LAUNCHED"] = "1"
    print(
        f"NUM_GPUS={config.NUM_GPUS} — re-launching under "
        f"`accelerate launch --num_processes={config.NUM_GPUS}` for real "
        f"data-parallel training instead of running single-process.",
        flush=True,
    )
    os.execvp(
        "accelerate",
        [
            "accelerate", "launch",
            "--num_processes", str(config.NUM_GPUS),
            "--num_machines", "1",
            "--mixed_precision", "fp16",
            __file__,
        ],
    )
    # os.execvp replaces this process entirely — nothing below this line
    # in this branch ever runs.
from scripts.data_loading import build_dataset_for_languages
from scripts.preprocessing import build_prepare_fn, WhisperDataCollator
from scripts.model_setup import load_processor, load_quantized_model, build_llrd_param_groups
from scripts.metrics import build_compute_metrics_fn
from scripts.training import build_trainer
from scripts.hf_push import authenticate, push_final_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_pipeline")


def main():
    logger.info(f"=== Starting run: {config.RUN_NAME} | languages: {config.LANGUAGES} ===")

    # 1. Authenticate early — fail fast if secrets are missing, before
    #    spending an hour on data loading / training.
    hf_token, hf_username = authenticate()

    # 2. Data
    logger.info("Loading and mixing datasets ...")
    dataset_dict = build_dataset_for_languages(config.LANGUAGES)

    # 3. Model + processor
    logger.info("Loading processor and quantized model ...")
    processor = load_processor()
    model = load_quantized_model()

    # 4. Preprocess (map over datasets)
    logger.info("Preprocessing train split (with augmentation) ...")
    train_prepare = build_prepare_fn(processor.feature_extractor, processor.tokenizer, is_training=True)
    train_dataset = dataset_dict["train"].map(
        train_prepare, remove_columns=dataset_dict["train"].column_names, num_proc=1
    )

    logger.info("Preprocessing validation split (no augmentation) ...")
    eval_prepare = build_prepare_fn(processor.feature_extractor, processor.tokenizer, is_training=False)
    eval_dataset = dataset_dict["validation"].map(
        eval_prepare, remove_columns=dataset_dict["validation"].column_names, num_proc=1
    )

    data_collator = WhisperDataCollator(processor=processor)
    compute_metrics = build_compute_metrics_fn(processor.tokenizer)

    # 5. LLRD param groups
    llrd_groups = build_llrd_param_groups(model)

    # 6. Trainer
    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        llrd_param_groups=llrd_groups,
        hf_username=hf_username,
        hf_token=hf_token,
    )

    logger.info("Starting training ...")
    trainer.train()

    logger.info("Training complete. Running final evaluation ...")
    final_metrics = trainer.evaluate()
    logger.info(f"Final validation WER: {final_metrics.get('eval_wer')}")

    # 7. Final push (separate from the periodic checkpoint pushes, tagged
    #    distinctly in the commit message so it's easy to find on HF).
    # Under accelerate launch --num_processes=2, this script runs once per
    # GPU process — guard so only the main process pushes, or you'd get two
    # processes racing to push the same final model.
    if trainer.is_world_process_zero():
        repo_id = push_final_model(model, processor, hf_username, hf_token)
        logger.info(f"=== Run {config.RUN_NAME} complete. Model: https://huggingface.co/{repo_id} ===")
    else:
        logger.info(f"=== Run {config.RUN_NAME} complete on this process (non-main, no push). ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed with an unhandled exception:")
        sys.exit(1)
