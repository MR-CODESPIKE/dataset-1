"""
Builds the Seq2SeqTrainer with:
  - the LLRD optimizer (Strategy 2) instead of Trainer's default single-LR AdamW
  - a callback that pushes a checkpoint to Hugging Face every config.SAVE_STEPS,
    so progress survives a Kaggle session dying mid-run
"""

import logging
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from scripts import config
from scripts.hf_push import push_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training")


class HFCheckpointCallback(TrainerCallback):
    """
    Pushes model + processor to Hugging Face Hub every save_steps.

    Under 2-GPU data-parallel training (accelerate launch --num_processes=2),
    Trainer runs this callback once per process. Without a guard, both GPUs'
    processes would try to push the same checkpoint simultaneously — wasted
    bandwidth at best, a corrupted/interleaved push at worst. `is_world_process_zero`
    restricts the push to a single (the "main") process regardless of NUM_GPUS.
    """

    def __init__(self, model, processor, username: str, token: str):
        self.model = model
        self.processor = processor
        self.username = username
        self.token = token

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        try:
            push_checkpoint(
                self.model, self.processor, state.global_step, self.username, self.token
            )
        except Exception as e:
            # Don't let a transient HF Hub error kill the whole training run —
            # local checkpoints (Kaggle's default Trainer output_dir) still exist.
            logger.error(f"HF checkpoint push failed at step {state.global_step}: {e}")


def build_training_args() -> Seq2SeqTrainingArguments:
    # NOTE on effective batch size under accelerate launch --num_processes=2:
    # per_device_train_batch_size is PER GPU. With NUM_GPUS=2, your real
    # effective batch size is:
    #   PER_DEVICE_BATCH_SIZE * GRAD_ACCUMULATION_STEPS * NUM_GPUS
    # i.e. it roughly doubles vs. a single-T4 run at the same config values.
    # This is expected and is exactly where the speedup comes from — no
    # action needed here, but be aware of it if you're tuning learning
    # rate/warmup against a specific effective batch size later.
    return Seq2SeqTrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION_STEPS,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=2,          # keep local disk usage bounded on Kaggle
        logging_steps=config.LOGGING_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,  # needed since we keep custom "lang"/"source" cols upstream
        label_names=["labels"],
    )


def build_trainer(
    model,
    processor,
    train_dataset,
    eval_dataset,
    data_collator,
    compute_metrics,
    llrd_param_groups,
    hf_username: str,
    hf_token: str,
) -> Seq2SeqTrainer:
    training_args = build_training_args()

    # LLRD: build a per-group optimizer instead of Trainer's default,
    # which would otherwise apply one flat LR to every parameter.
    optimizer = torch.optim.AdamW(llrd_param_groups)

    from transformers import get_linear_schedule_with_warmup

    num_training_steps = (
        len(train_dataset)
        // (config.PER_DEVICE_BATCH_SIZE * config.GRAD_ACCUMULATION_STEPS)
        * config.NUM_TRAIN_EPOCHS
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[HFCheckpointCallback(model, processor, hf_username, hf_token)],
    )

    return trainer
