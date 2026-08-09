"""
Handles authentication and pushing checkpoints / the final fine-tuned
model to Hugging Face Hub.

Expects two Kaggle Secrets to be set (Kaggle notebook -> Add-ons -> Secrets):
    HF_TOKEN     - a Hugging Face WRITE token (huggingface.co/settings/tokens)
    HF_USERNAME  - your HF username, used to build the target repo id

On Kaggle, secrets are retrieved via kaggle_secrets.UserSecretsClient, NOT
plain os.environ — see get_kaggle_secret() below. Falls back to os.environ
for local/Colab use.
"""

import logging
import os

from huggingface_hub import login, HfApi

from scripts import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf_push")


def get_secret(name: str) -> str:
    """Fetch a secret from Kaggle Secrets if available, else from env vars."""
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        return client.get_secret(name)
    except Exception:
        value = os.environ.get(name)
        if value is None:
            raise RuntimeError(
                f"Could not find secret '{name}' in Kaggle Secrets or "
                f"environment variables. Set it before running."
            )
        return value


def authenticate() -> tuple[str, str]:
    token = get_secret(config.HF_TOKEN_ENV)
    username = get_secret(config.HF_USERNAME_ENV)
    login(token=token)
    logger.info(f"Authenticated to Hugging Face Hub as '{username}'")
    return token, username


def get_repo_id(username: str) -> str:
    return f"{username}/{config.HF_REPO_NAME_TEMPLATE.format(run_name=config.RUN_NAME)}"


def push_checkpoint(model, processor, step: int, username: str, token: str):
    """
    Called periodically during training (see training.py's callback).
    Pushes to a branch/commit-message tagged by step so intermediate
    checkpoints are recoverable if a Kaggle session dies mid-run, without
    overwriting the repo's main history each time.
    """
    repo_id = get_repo_id(username)
    commit_message = f"checkpoint at step {step} ({config.RUN_NAME})"
    logger.info(f"Pushing checkpoint (step {step}) to {repo_id} ...")
    model.push_to_hub(repo_id, token=token, commit_message=commit_message)
    processor.push_to_hub(repo_id, token=token, commit_message=commit_message)
    logger.info("Checkpoint push complete.")


def push_final_model(model, processor, username: str, token: str):
    repo_id = get_repo_id(username)
    logger.info(f"Pushing FINAL model to {repo_id} ...")
    model.push_to_hub(
        repo_id, token=token, commit_message=f"final model — {config.RUN_NAME}"
    )
    processor.push_to_hub(
        repo_id, token=token, commit_message=f"final model — {config.RUN_NAME}"
    )
    logger.info(f"Done. Model available at https://huggingface.co/{repo_id}")
    return repo_id
