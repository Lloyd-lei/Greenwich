"""Upload staged weights to the HF hub (lloyd-lei/Greenwich)."""
from pathlib import Path

from huggingface_hub import HfApi

STAGE = Path(__file__).parent.parent / "weights_staging"
REPO = "lloyd-lei/Greenwich"

api = HfApi()
api.create_repo(REPO, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=str(STAGE), repo_id=REPO,
                  commit_message="AlphaMotion v0.1 release artifacts")
print(f"uploaded {REPO}")
