#!/usr/bin/env python3
"""
run_inference.py — batched vLLM inference over base + transformed prompt sets
for one selected environment.

Run order (sequential, single env):
    1. base harmful          (all)
    2. base safe             (subsample n = |base_harmful|)
    3. transformed harmful   (all records, full fan-out)
    4. transformed safe      (records whose orig_index in the sampled-safe set)

Outputs:
    outputs/responses/<env>/base/harmful.json
    outputs/responses/<env>/base/safe.json
    outputs/responses/<env>/transformed/harmful.json
    outputs/responses/<env>/transformed/safe.json
    outputs/responses/<env>/sampled_indices.json   (provenance)

Entry schema:
    {
      "prompt": <original base prompt str>,
      "transformed_prompt": <str or null>,   # null for base runs
      "orig_index": <int>,                   # index into base data file
      "response": <model completion str>,
      "refusal": null                        # filled later by Llama-Guard-4
    }

Not automated across envs by design: pass --env to pick one.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Optional
from vllm import LLM, SamplingParams
from src.config import DATA_DIR, MODEL_DIR, OUTPUT_DIR


# ── paths / defaults (from config.py) ───────────────────────────────
TRANSFORMED_DIR = DATA_DIR / "transformed"
OUT_ROOT = OUTPUT_DIR / "responses"
DEFAULT_MODEL = MODEL_DIR

# env list stays local: config.HARMFUL_ENVIRONMENTS uses stale names
# (nsfw/terrorism/hate_harassment_violence) that don't match the actual
# data files written by transform.py.
ENVIRONMENTS = [
    "cybersecurity", "medical", "hate_harassment",
    "general_crime", "bioterrorism",
]


# ── IO helpers ──────────────────────────────────────────────────────
def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_base(env: str, split: str) -> List[str]:
    """Base prompt list (index = position in file)."""
    return load_json(DATA_DIR / f"{env}_{split}.json")


def load_transformed(env: str, split: str) -> List[dict]:
    """Transformed records: {env, split, orig_index, transform, variant_index, text}."""
    return load_json(TRANSFORMED_DIR / f"{env}_{split}_transformed.json")


# ── prompt formatting ───────────────────────────────────────────────
def to_chat(tokenizer, user_text: str) -> str:
    """Render single user turn through the model's chat template."""
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── core batched generation ─────────────────────────────────────────
def generate(llm: LLM, sampling: SamplingParams, prompts: List[str]) -> List[str]:
    """Batched vLLM generation. Output order matches input order."""
    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text for o in outputs]


def build_entries(
    base_prompts: List[str],
    transformed_texts: Optional[List[str]],
    orig_indices: List[int],
    responses: List[str],
) -> List[dict]:
    """Assemble output entries with the fixed schema."""
    entries = []
    for i, (oi, resp) in enumerate(zip(orig_indices, responses)):
        entries.append({
            "prompt": base_prompts[i],
            "transformed_prompt": (
                transformed_texts[i] if transformed_texts is not None else None
            ),
            "orig_index": oi,
            "response": resp,
            "refusal": None,
        })
    return entries


# ── per-set runners ─────────────────────────────────────────────────
def run_base_split(
    llm, sampling, tokenizer, env: str, split: str,
    sample_indices: Optional[List[int]] = None,
) -> List[dict]:
    """Run a base split. If sample_indices given, restrict to those indices.

    Non-string rows (pandas NaN) are skipped and never sampled.
    """
    raw = load_base(env, split)
    valid = [(i, p) for i, p in enumerate(raw) if isinstance(p, str)]

    if sample_indices is not None:
        keep = set(sample_indices)
        valid = [(i, p) for (i, p) in valid if i in keep]

    orig_indices = [i for i, _ in valid]
    base_prompts = [p for _, p in valid]

    rendered = [to_chat(tokenizer, p) for p in base_prompts]
    responses = generate(llm, sampling, rendered)

    return build_entries(base_prompts, None, orig_indices, responses)


def run_transformed_split(
    llm, sampling, tokenizer, env: str, split: str,
    base_raw: List[str],
    keep_indices: Optional[List[int]] = None,
) -> List[dict]:
    """Run a transformed split (full fan-out).

    If keep_indices given (transformed_safe), restrict to records whose
    orig_index is in that set. `base_raw` recovers the original prompt string
    for each record's orig_index.
    """
    records = load_transformed(env, split)
    if keep_indices is not None:
        keep = set(keep_indices)
        records = [r for r in records if r["orig_index"] in keep]

    transformed_texts = [r["text"] for r in records]
    orig_indices = [r["orig_index"] for r in records]
    base_prompts = [base_raw[oi] for oi in orig_indices]

    rendered = [to_chat(tokenizer, t) for t in transformed_texts]
    responses = generate(llm, sampling, rendered)

    return build_entries(base_prompts, transformed_texts, orig_indices, responses)


# ── main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=ENVIRONMENTS,
                        help="environment to run inference for")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="path to local HF model dir for vLLM")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for safe-set subsampling (reproducible)")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy; deterministic responses")
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="cap context; artprompt/dan prompts can be long")
    parser.add_argument("--dtype", default="float16",
                        help="float16 for Pascal (GTX 1080 Ti); bf16 unsupported")
    # add to argparse block
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM TP shards across GPUs (4 for 4x GTX 1080 Ti)")
    args = parser.parse_args()

    env = args.env
    out_dir = OUT_ROOT / env

    # ── load model once ─────────────────────────────────────────────
    print(f"Loading vLLM model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # ── n = |base_harmful| (valid strings only) ─────────────────────
    base_harmful_raw = load_base(env, "harmful")
    base_safe_raw = load_base(env, "safe")

    n_harmful = sum(1 for p in base_harmful_raw if isinstance(p, str))
    print(f"[{env}] |base_harmful| (valid) = {n_harmful}")

    # ── subsample safe indices: n = |base_harmful|, reproducible ────
    valid_safe_indices = [
        i for i, p in enumerate(base_safe_raw) if isinstance(p, str)
    ]
    rng = random.Random(args.seed)
    n_safe = min(n_harmful, len(valid_safe_indices))
    sampled_safe_indices = sorted(rng.sample(valid_safe_indices, n_safe))
    print(f"[{env}] sampled {n_safe} safe indices "
          f"(from {len(valid_safe_indices)} valid)")

    # provenance sidecar — exact indices sampled from the safe set
    write_json(out_dir / "sampled_indices.json", {
        "env": env,
        "seed": args.seed,
        "n_harmful": n_harmful,
        "n_safe_sampled": n_safe,
        "sampled_safe_indices": sampled_safe_indices,
    })

    # ── 1. base harmful (all) ───────────────────────────────────────
    print(f"[{env}] 1/4 base harmful ...")
    e = run_base_split(llm, sampling, tokenizer, env, "harmful")
    write_json(out_dir / "base" / "harmful.json", e)
    print(f"        wrote {len(e)} entries")

    # ── 2. base safe (subsampled) ───────────────────────────────────
    print(f"[{env}] 2/4 base safe (subsampled) ...")
    e = run_base_split(llm, sampling, tokenizer, env, "safe",
                       sample_indices=sampled_safe_indices)
    write_json(out_dir / "base" / "safe.json", e)
    print(f"        wrote {len(e)} entries")

    # ── 3. transformed harmful (full fan-out) ───────────────────────
    print(f"[{env}] 3/4 transformed harmful (full fan-out) ...")
    e = run_transformed_split(llm, sampling, tokenizer, env, "harmful",
                              base_raw=base_harmful_raw)
    write_json(out_dir / "transformed" / "harmful.json", e)
    print(f"        wrote {len(e)} entries")

    # ── 4. transformed safe (same base indices, all variants) ───────
    print(f"[{env}] 4/4 transformed safe (sampled indices, all variants) ...")
    e = run_transformed_split(llm, sampling, tokenizer, env, "safe",
                              base_raw=base_safe_raw,
                              keep_indices=sampled_safe_indices)
    write_json(out_dir / "transformed" / "safe.json", e)
    print(f"        wrote {len(e)} entries")

    print(f"[{env}] done. Outputs under {out_dir}")


if __name__ == "__main__":
    main()