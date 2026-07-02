from src.config import hf_token, BASE_DIR
from huggingface_hub import login, snapshot_download

print(BASE_DIR)

login(token=hf_token)

REPO_ID = "meta-llama/Llama-3.2-3B-Instruct"
snapshot_download(repo_id=REPO_ID, local_dir=f"{BASE_DIR}/models/Llama-3.2-3B-Instruct")