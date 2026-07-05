from src.config import hf_token, BASE_DIR
from huggingface_hub import login, snapshot_download

print(BASE_DIR)

login(token=hf_token)

REPO_ID = "meta-llama/Llama-Guard-4-12B"
snapshot_download(repo_id=REPO_ID, local_dir=f"{BASE_DIR}/models/meta-llama/Llama-Guard-4-12B")