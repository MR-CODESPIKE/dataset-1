"""
Loads Sunbird Whisper-51 in 4-bit NF4 (QLoRA), wraps it with LoRA adapters,
and builds the Layer-Wise Learning Rate Decay (Strategy 2) optimizer param
groups: low LR on the audio encoder, higher LR on the LoRA adapters that
sit in the decoder's attention modules.
"""

import logging
import os
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_setup")


def load_processor() -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(config.BASE_MODEL_ID)


def load_quantized_model() -> WhisperForConditionalGeneration:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading {config.BASE_MODEL_ID} in 4-bit NF4 ...")

    if config.NUM_GPUS > 1:
        # IMPORTANT: device_map="auto" would SPLIT one model's layers across
        # both GPUs (memory headroom, no speedup). For real 2-GPU speedup we
        # instead want each process to hold a FULL model copy on its own
        # single GPU, and let `accelerate launch --num_processes=2` handle
        # data parallelism (different batches per GPU, gradients synced).
        # So here we pin this process to its own assigned GPU via the
        # LOCAL_RANK env var that `accelerate launch` sets automatically —
        # do NOT pass device_map="auto" in this branch.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        logger.info(
            f"NUM_GPUS={config.NUM_GPUS} — loading one full model copy on "
            f"local rank {local_rank} for data-parallel training via accelerate."
        )
    else:
        device_map = "auto"

    model = WhisperForConditionalGeneration.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model.config.use_cache = False  # required for gradient checkpointing during training

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def build_llrd_param_groups(model) -> list[dict]:
    """
    Strategy 2: Layer-Wise Learning Rate Decay.

    - Whisper encoder blocks (already strong at hearing raw audio, from
      Sunbird's own African-language pretraining) get a very small LR.
    - LoRA adapter params — which live in the decoder's q_proj/v_proj
      attention modules per config.LORA_TARGET_MODULES — get a much
      higher LR, since that's where the actual language adaptation to
      Lingala/Shona/Luganda vocabulary needs to happen.
    - Any other trainable param not caught by the two rules above falls
      back to the decoder adapter LR, on the assumption it's more likely
      decoder-side than encoder-side.
    """
    encoder_params = []
    decoder_adapter_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            encoder_params.append(param)
        elif "lora" in name.lower():
            decoder_adapter_params.append(param)
        else:
            other_params.append(param)

    logger.info(
        f"LLRD groups — encoder: {len(encoder_params)} tensors "
        f"@ {config.ENCODER_LR}, decoder/LoRA: {len(decoder_adapter_params)} "
        f"tensors @ {config.DECODER_ADAPTER_LR}, other: {len(other_params)} tensors"
    )

    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": config.ENCODER_LR})
    if decoder_adapter_params:
        groups.append({"params": decoder_adapter_params, "lr": config.DECODER_ADAPTER_LR})
    if other_params:
        groups.append({"params": other_params, "lr": config.DECODER_ADAPTER_LR})

    return groups
