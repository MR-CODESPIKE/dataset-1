"""
Central configuration for the WAXAL ASR fine-tuning pipeline.
Edit values here rather than hunting through other files.
"""

import os

# ---------------------------------------------------------------------------
# Run identity — CHANGE run_name per run (e.g. "sna_lug_combined", "lin_solo")
# ---------------------------------------------------------------------------
RUN_NAME = os.environ.get("WAXAL_RUN_NAME", "sna_lug_combined")

# Which languages this run trains on. Use WaxalNLP config codes.
# Combined run: ["sna_asr", "lug_asr"]
# Lingala solo run: ["lin_asr"]
LANGUAGES = os.environ.get("WAXAL_LANGUAGES", "sna_asr,lug_asr").split(",")

# ISO codes used to pick the right FLEURS / AfriVoice / CommonVoice configs
# per language. Extend this dict if you add more supplementary datasets.
LANG_ISO = {
    "sna_asr": {"fleurs": "sn_zw", "afrivoice": "sn", "common_voice": "sn", "waxal_wer_baseline": 0.196},
    "lug_asr": {"fleurs": "lg_ug", "afrivoice": None,  "common_voice": "lg", "waxal_wer_baseline": 0.109},
    "lin_asr": {"fleurs": "ln_cd", "afrivoice": "ln",  "common_voice": None, "waxal_wer_baseline": 0.243},
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "Sunbird/asr-whisper-51-african-languages"

# ---------------------------------------------------------------------------
# Supplementary open-license datasets to mix in alongside WaxalNLP.
# Toggle individually — start with False, turn on once WaxalNLP-only baseline works.
# ---------------------------------------------------------------------------
USE_FLEURS = True
USE_AFRIVOICE = True
USE_COMMON_VOICE = False  # requires HF login + dataset-specific agreement, enable once confirmed

# ---------------------------------------------------------------------------
# Paths (Kaggle working directory persists for the session / becomes kernel output)
# ---------------------------------------------------------------------------
WORK_DIR = os.environ.get("WAXAL_WORK_DIR", "/kaggle/working")
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints", RUN_NAME)
LOG_DIR = os.path.join(WORK_DIR, "logs", RUN_NAME)

# ---------------------------------------------------------------------------
# Hugging Face push targets
# ---------------------------------------------------------------------------
# Set these as Kaggle Secrets: HF_TOKEN, HF_USERNAME
HF_TOKEN_ENV = "HF_TOKEN"
HF_USERNAME_ENV = "HF_USERNAME"
HF_REPO_NAME_TEMPLATE = "waxal-whisper-{run_name}"  # -> {username}/waxal-whisper-sna_lug_combined
# NOTE: LANGID routing above hardcodes the run names "sna_lug_combined" and
# "lin_solo" — if you change RUN_NAME for those training runs, update
# WHISPER_LANGID_TO_RUN and LANGID_FALLBACK_RUN above to match, since
# inference.py resolves HF repo ids from these exact strings.

# ---------------------------------------------------------------------------
# Hardware — Kaggle "GPU T4 x2" gives ONE session with 2 physical T4s.
# This is real data-parallel training via `accelerate launch`, not just
# bigger device_map memory headroom. See training.py / run_pipeline.py.
# ---------------------------------------------------------------------------
NUM_GPUS = int(os.environ.get("WAXAL_NUM_GPUS", "2"))

# ---------------------------------------------------------------------------
# Phase 2 language identification.
# Test_Phase2.csv confirmed: IDs are opaque (ID_XXXXXX), no language prefix,
# unlike Phase 1's lang-prefixed IDs (lug_/lin_/sna_). So which fine-tuned
# checkpoint (sna_lug_combined vs lin_solo) to use per clip is NOT known
# ahead of time and must be predicted from audio at inference time.
# ---------------------------------------------------------------------------
LANGID_ENABLED = True
# "whisper" = use the base Whisper model's own built-in language detection
#             (cheap, one forward pass, no extra model to train/host)
# "all_checkpoints" = run every fine-tuned checkpoint on every clip and
#             pick the most confident output (slower, no extra model needed
#             either, but ~2-3x inference compute)
LANGID_METHOD = os.environ.get("WAXAL_LANGID_METHOD", "whisper")

# Whisper's language detection returns generic ISO 639-1 codes (e.g. "ln",
# "sn", "lg"), not WaxalNLP's config-style codes — this maps between them.
WHISPER_LANGID_TO_RUN = {
    "ln": "lin_solo",           # Lingala -> the dedicated Lingala checkpoint
    "sn": "sna_lug_combined",   # Shona    -> the combined checkpoint
    "lg": "sna_lug_combined",   # Luganda  -> the combined checkpoint
}
# Fallback run if Whisper's detected language isn't one of the three above
# (e.g. Phase 2 genuinely includes a language outside your training set —
# a real possibility per the Zindi discussion thread on this).
LANGID_FALLBACK_RUN = "sna_lug_combined"

# ---------------------------------------------------------------------------
# QLoRA / LoRA
# ---------------------------------------------------------------------------
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ---------------------------------------------------------------------------
# Layer-Wise Learning Rate Decay (Strategy 2)
# ---------------------------------------------------------------------------
ENCODER_LR = 1e-6
DECODER_ADAPTER_LR = 2e-4

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 8       # reduce to 4 if you hit CUDA OOM on the T4
GRAD_ACCUMULATION_STEPS = 2     # effective batch size = PER_DEVICE_BATCH_SIZE * this
SAVE_STEPS = 200                # checkpoint (local + HF push) every N steps
EVAL_STEPS = 200
LOGGING_STEPS = 25
WARMUP_STEPS = 50
MAX_AUDIO_SECONDS = 30          # Whisper's native window; longer clips get truncated in preprocessing

# ---------------------------------------------------------------------------
# Data Augmentation (Strategy 1)
# ---------------------------------------------------------------------------
AUGMENT_ENABLED = True
SPEED_PERTURB_RANGE = (0.9, 1.1)
SPEED_PERTURB_PROB = 0.3
SPECAUGMENT_FREQ_MASK_PARAM = 15
SPECAUGMENT_TIME_MASK_PARAM = 35
SPECAUGMENT_NUM_MASKS = 2
SPECAUGMENT_PROB = 0.3

# Give Lingala extra augmentation weight since it's the weakest baseline —
# only applies when "lin_asr" is in LANGUAGES for this run.
LINGALA_AUGMENT_BOOST = 1.5

# ---------------------------------------------------------------------------
# Text normalization (Strategy 4) — tune once you've inspected real transcripts
# ---------------------------------------------------------------------------
NORMALIZE_LOWERCASE = True
NORMALIZE_STRIP_PUNCTUATION = True
