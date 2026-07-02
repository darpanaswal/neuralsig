import os
import numpy as np
import random, torch, os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


BASE_DIR = Path(__file__).parent.parent
dotenv_path = BASE_DIR / ".env"

load_dotenv(dotenv_path=dotenv_path)

openai_token = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")

if not openai_token or not hf_token or not wandb_token:
    raise ValueError("API keys are not set in environment variables")

# ─────────────────────────── Paths ─────────────────────────────────
DATA_DIR = BASE_DIR / "data"
HARMFUL_ENVIRONMENTS = [
    "cybersecurity",
    "hate_harassment_violence",
    "medical",
    "nsfw",
    "terrorism",
]

MODEL_DIR = BASE_DIR / "model/meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = BASE_DIR / "outputs"