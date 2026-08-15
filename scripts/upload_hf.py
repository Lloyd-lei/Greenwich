"""Upload staged weights to the HF hub (lloyd-lei/Greenwich)."""
from pathlib import Path

from huggingface_hub import HfApi

STAGE = Path(__file__).parent.parent / "weights_staging"
REPO = "lloydlei/Greenwich"

import os

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
api.create_repo(REPO, exist_ok=True, repo_type="model")
info = api.upload_folder(folder_path=str(STAGE), repo_id=REPO,
                         commit_message="AlphaMotion v0.1 release artifacts")
print("commit:", info.commit_url if hasattr(info, "commit_url") else info)
files = api.list_repo_files(REPO)
print(f"uploaded {REPO}: {len(files)} files")
for f in sorted(files):
    print("  ", f)
