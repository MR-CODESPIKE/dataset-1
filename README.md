# WAXAL ASR Fine-tuning Pipeline

Fine-tunes `Sunbird/asr-whisper-51-african-languages` for Lingala, Shona,
and Luganda ASR using QLoRA, with 4 strategies: data augmentation, LLRD,
KenLM shallow fusion, and text normalization.

## One-time setup (do this before your first push)

1. **Kaggle account + API token**: Kaggle Settings → API → Create New Token.
   Add as GitHub repo secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`.
2. **Hugging Face WRITE token**: huggingface.co/settings/tokens. This does
   NOT go in GitHub secrets — it goes in **Kaggle Secrets** instead, since
   it's used inside the Kaggle notebook, not by the GitHub Actions runner.
   Kaggle notebook → Add-ons → Secrets → add `HF_TOKEN` and `HF_USERNAME`.
3. **Accept AfriVoice's gated dataset terms** (one-time, in a browser):
   https://huggingface.co/datasets/DigitalUmuganda/AfriVoice
4. Edit `kernel-metadata.json`: replace `YOUR_KAGGLE_USERNAME`.
5. Before trusting `text_normalize.py`'s defaults, run it standalone once to
   inspect real transcripts:
   ```
   python scripts/text_normalize.py
   ```
   Confirm WaxalNLP transcripts are actually lowercase/unpunctuated as
   assumed — adjust `config.py`'s `NORMALIZE_*` flags if not.

## Running training

Two separate runs, per the combined-then-Lingala plan:

**Run 1 — Shona + Luganda combined** (validates the whole pipeline on the
stronger-baseline languages first):
- GitHub → Actions → "Deploy WAXAL pipeline to Kaggle" → Run workflow
- `run_name`: `sna_lug_combined`
- `languages`: `sna_asr,lug_asr`

**Run 2 — Lingala solo** (fresh fine-tune from the Sunbird base, NOT
continued from Run 1's checkpoint):
- Same workflow, new inputs:
- `run_name`: `lin_solo`
- `languages`: `lin_asr`

## What's automated vs. manual

| Step | Automated? |
|---|---|
| Push code to Kaggle, start GPU session | ✅ via GitHub Actions |
| Data loading, augmentation, training, checkpointing | ✅ runs unattended on Kaggle |
| Pushing checkpoints + final model to Hugging Face | ✅ happens inside the training script |
| **Checking if a run is still going / finished / crashed** | ✅ Partially — `kaggle kernels status <id>` is a real command (correcting an earlier claim in this project that it wasn't possible); the workflow now polls it every 5 min. Caveat: some users report a "kernels.get permission denied" error on this endpoint, and a GitHub Actions job itself has a ~6hr runtime cap — for a training run that outlives that, still check kaggle.com/code manually |
| **Submitting to Zindi** | ❌ manual — Zindi has no confirmed automation path |

## Sample submission (format-test file included)

`Test_Phase2.csv` as downloaded from Zindi IS a valid placeholder submission
(every `Target` is `"the quick brown fox"`) — it's their own
`SampleSubmission.csv` with real IDs. Included here as
`SampleSubmission_Phase2.csv` so you can submit it early purely to confirm
your Zindi submission flow works, before a real model exists.

## After training: inference + submission

Run `scripts/inference.py` interactively (separate Kaggle session or
locally if you have a GPU) once a model is on Hugging Face:
1. Confirm the real Zindi Phase 2 test CSV column names first.
2. `train_kenlm()` is a one-time manual step — see its docstring.
3. `build_submission()` writes the final CSV in Zindi's expected format.

## Real training data counts (confirmed from Zindi's Train.csv)

Inspected directly (38,176 clean rows via proper CSV quote handling — see
`data_loading.load_waxal_csv_transcripts()`; pandas' default parser drops
~150 rows on this file due to embedded quotes/commas in some transcripts):

| Language | Rows | Baseline WER (Sunbird, pre-finetune) |
|---|---|---|
| Lingala | 16,240 | 24.3% |
| Shona | 15,817 | 19.6% |
| Luganda | 6,119 | 10.9% |

**Correction to earlier planning:** Lingala was assumed to be the
data-scarce language (hence `LINGALA_AUGMENT_BOOST` in config.py) — it's
actually the largest split. Luganda has the least data but the best WER,
suggesting Sunbird's own pretraining already covers it well and your
fine-tuning has less headroom there. Worth reconsidering training epoch
allocation between Shona and Luganda in the combined run given this.

**Real transcript format confirmed:** mixed case, full punctuation (e.g.
`"Ekyuma ekyakolebwa Bamagulumeeru nga kiri mu makkati g'ennyanja..."`).
`config.py`'s `NORMALIZE_LOWERCASE` / `NORMALIZE_STRIP_PUNCTUATION` are now
set to `False` to match — this was previously a guess, now corrected
against real data.

## Dual T4 (Kaggle "GPU T4 x2")

Confirmed via the accelerator dropdown: this is ONE session with 2 physical
T4s, not 2 separate sessions. `run_pipeline.py` now self-relaunches under
`accelerate launch --num_processes=2` on Kaggle (Kaggle's kernel runner
calls `python run_pipeline.py` directly with no way to change that command,
so the script re-execs itself via `os.execvp`). `model_setup.py` loads one
full model copy per GPU process instead of splitting one model across both
(which is what plain `device_map="auto"` would have done — memory headroom,
not speed). HF checkpoint/final pushes are guarded to the main process only,
so both GPUs don't race to push the same commit. Set `WAXAL_NUM_GPUS=1` if
you ever run this on a single-GPU box instead.

## Phase 2 language routing (confirmed from real Test.csv / Test_Phase2.csv)

- **Phase 1** (`Test.csv`): IDs are language-prefixed (`lug_96114`,
  `lin_24783`, `sna_55007`) — `inference.py` reads the prefix directly, no
  detection needed.
- **Phase 2** (`Test_Phase2.csv`): IDs are opaque (`ID_QNYPTX`, columns
  `ID,Target`) — confirmed no language prefix. `inference.py` falls back to
  Whisper's own built-in `detect_language()` per clip, then routes to the
  matching fine-tuned checkpoint (`sna_lug_combined` or `lin_solo`). Both
  checkpoints are loaded for inference regardless of phase, via
  `load_all_checkpoints()`.
- **Now implemented:** `load_test_audio()` in `inference.py` auto-detects
  whichever delivery format Zindi actually used — a directory of per-ID
  audio files, a `.zip`/`.tar` archive (auto-extracted), or a Hugging Face
  dataset repo id — and raises a specific, actionable error if none match,
  rather than guessing. Point `audio_source` at whatever you find on
  Zindi's Data page once downloaded.

## Known open items (confirm before Day 3)

- How Phase 2 audio is delivered/joined to Test_Phase2.csv's IDs (see above).
- KenLM shallow fusion for Whisper (seq2seq, not CTC) is implemented here
  as an n-best rescoring TODO, not per-step fusion — see `inference.py`
  docstring for why, and finish that piece once base inference is working.
