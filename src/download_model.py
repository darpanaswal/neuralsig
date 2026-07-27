import os

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from src.config import hf_token, BASE_DIR
from huggingface_hub import login, snapshot_download

print(BASE_DIR)

login(token=hf_token)

REPO_ID = "google/gemma-1.1-7b-it"
snapshot_download(repo_id=REPO_ID, local_dir=f"{BASE_DIR}/models/google/gemma-1.1-7b-it")