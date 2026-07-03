import json
from pathlib import Path
from safetensors.torch import load_file, save_file

model_dir = Path("models/meta-llama/Llama-3.2-3B-Instruct")
shard1 = model_dir / "model-00001-of-00002.safetensors"
lm_head_file = model_dir / "lm_head.safetensors"
index_file = model_dir / "model.safetensors.index.json"

print("Loading embed_tokens...")
tensors = load_file(shard1)
embed_weights = tensors["model.embed_tokens.weight"]

print("Saving lm_head (this takes a few seconds)...")
save_file({"lm_head.weight": embed_weights}, lm_head_file)

print("Patching index.json...")
with open(index_file, "r") as f:
    index = json.load(f)

index["weight_map"]["lm_head.weight"] = "lm_head.safetensors"

with open(index_file, "w") as f:
    json.dump(index, f, indent=2)

print("Done! You have successfully reattached the model's head.")