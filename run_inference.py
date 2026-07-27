#!/usr/bin/env python3
"""
run_inference.py — batched vLLM inference over base + transformed prompt sets
for one selected environment.

Each jailbreak type produced by jailbreak_transforms/transform.py lives in
its own folder (data/transformed/<type>/<env>_<split>.json, e.g. dan/,
opposite_mode/, payload_split/, ...); this script discovers those folders
and runs inference over every one of them, in addition to the base sets.

Run order (sequential, single env, per shard):
    1. base harmful                    (all, sharded)
    2. base safe                       (all, sharded)
    3. per jailbreak type: harmful     (all, sharded)
    4. per jailbreak type: safe        (all, sharded)

Outputs:
    outputs/responses/<env>/base/harmful_<shard_id>.json
    outputs/responses/<env>/base/safe_<shard_id>.json
    outputs/responses/<env>/transformed/<type>/harmful_<shard_id>.json
    outputs/responses/<env>/transformed/<type>/safe_<shard_id>.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional
from vllm import LLM, SamplingParams
from src.config import DATA_DIR, LLAMA2, LLAMA3, MISTRAL, GEMMA, OUTPUT_DIR


# ── paths / defaults (from config.py) ───────────────────────────────
TRANSFORMED_DIR = DATA_DIR / "transformed"
OUT_ROOT = OUTPUT_DIR / "responses"

MODELS = ["llama2", "llama3", "mistral", "gemma"]

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
    return load_json(DATA_DIR / f"{env}_{split}.json")


# bon_augment's random word-scrambling/noise produces too many garbled,
# semantically-unrelated prompts to be useful for probe training - excluded
# from generation and every downstream stage (guard, judge, annotation).
EXCLUDED_JAILBREAK_TYPES = {"bon_augment"}


def discover_jailbreak_types(env: str, split: str) -> List[str]:
    """Jailbreak-type subfolders under TRANSFORMED_DIR with data for this
    env/split, e.g. dan, opposite_mode, payload_split, ..."""
    if not TRANSFORMED_DIR.exists():
        return []
    return sorted(
        d.name for d in TRANSFORMED_DIR.iterdir()
        if d.is_dir() and d.name not in EXCLUDED_JAILBREAK_TYPES
        and (d / f"{env}_{split}.json").exists()
    )


def load_transformed(env: str, split: str, jb_type: str) -> List[dict]:
    return load_json(TRANSFORMED_DIR / jb_type / f"{env}_{split}.json")


# ── prompt formatting ───────────────────────────────────────────────
def to_chat(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── core batched generation ─────────────────────────────────────────
def generate(llm: LLM, sampling: SamplingParams, prompts: List[str]) -> List[str]:
    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text for o in outputs]


def build_entries(
    base_prompts: List[str],
    transformed_texts: Optional[List[str]],
    orig_indices: List[int],
    responses: List[str],
) -> List[dict]:
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
    keep_indices: Optional[List[int]] = None,
    debug: bool = False,
) -> List[dict]:
    raw = load_base(env, split)
    valid = [(i, p) for i, p in enumerate(raw) if isinstance(p, str)]

    if keep_indices is not None:
        keep = set(keep_indices)
        valid = [(i, p) for (i, p) in valid if i in keep]

    if debug:
        valid = valid[:2]

    orig_indices = [i for i, _ in valid]
    base_prompts = [p for _, p in valid]

    rendered = [to_chat(tokenizer, p) for p in base_prompts]
    responses = generate(llm, sampling, rendered)

    if debug:
        print(f"\n{'='*20} DEBUG: BASE {split.upper()} {'='*20}")
        for p, r in zip(base_prompts, responses):
            print(f"PROMPT: {p}\nRESPONSE: {r}\n{'-'*50}")
        print("="*60 + "\n")

    return build_entries(base_prompts, None, orig_indices, responses)


def run_transformed_split(
    llm, sampling, tokenizer, env: str, split: str, jb_type: str,
    base_raw: List[str],
    keep_indices: Optional[List[int]] = None,
    debug: bool = False,
) -> List[dict]:
    records = load_transformed(env, split, jb_type)
    if keep_indices is not None:
        keep = set(keep_indices)
        records = [r for r in records if r["orig_index"] in keep]

    if debug:
        records = records[:2]

    transformed_texts = [r["text"] for r in records]
    orig_indices = [r["orig_index"] for r in records]
    base_prompts = [base_raw[oi] for oi in orig_indices]

    rendered = [to_chat(tokenizer, t) for t in transformed_texts]
    responses = generate(llm, sampling, rendered)

    if debug:
        print(f"\n{'='*20} DEBUG: {jb_type.upper()} {split.upper()} {'='*20}")
        for t, r in zip(transformed_texts, responses):
            print(f"TRANSFORMED PROMPT: {t}\nRESPONSE: {r}\n{'-'*50}")
        print("="*65 + "\n")

    return build_entries(base_prompts, transformed_texts, orig_indices, responses)


# ── main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=ENVIRONMENTS,
                        help="environment to run inference for")
    parser.add_argument("--model", required=True, choices=MODELS,
                        help="model to run inference on")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of data-parallel workers")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="The index (0 to num_shards-1) for this worker")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--debug", action="store_true",
                        help="Run only 2 samples per split")
    args = parser.parse_args()

    env = args.env
    out_dir = OUT_ROOT / f"{args.model}/{env}"
    if args.model == "llama2":
        args.model = LLAMA2
    elif args.model == "llama3":
        args.model = LLAMA3
    elif args.model == "mistral":
        args.model = MISTRAL
    elif args.model == "gemma":
        args.model = GEMMA

    print(f"Loading vLLM model: {args.model} on Shard {args.shard_id}/{args.num_shards}")
    # Force tensor_parallel_size=1 since we are scaling via Data Parallelism instead
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=1, 
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=True, 
        swap_space=0, 
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Load base raw to extract valid indices
    base_harmful_raw = load_base(env, "harmful")
    base_safe_raw = load_base(env, "safe")

    valid_harmful_indices = [i for i, p in enumerate(base_harmful_raw) if isinstance(p, str)]
    valid_safe_indices = [i for i, p in enumerate(base_safe_raw) if isinstance(p, str)]

    # Slice indices according to shard id (interleaved sharding)
    my_harmful_indices = valid_harmful_indices[args.shard_id :: args.num_shards]
    my_safe_indices = valid_safe_indices[args.shard_id :: args.num_shards]

    print(f"[{env} | Shard {args.shard_id}] Processing {len(my_harmful_indices)} harmful, {len(my_safe_indices)} safe prompts.")

    # 1. base harmful
    print(f"[{env} | Shard {args.shard_id}] base harmful ...")
    e = run_base_split(llm, sampling, tokenizer, env, "harmful", keep_indices=my_harmful_indices, debug=args.debug)
    write_json(out_dir / "base" / f"harmful_{args.shard_id}.json", e)

    # 2. base safe
    print(f"[{env} | Shard {args.shard_id}] base safe ...")
    e = run_base_split(llm, sampling, tokenizer, env, "safe", keep_indices=my_safe_indices, debug=args.debug)
    write_json(out_dir / "base" / f"safe_{args.shard_id}.json", e)

    # 3/4. one pass per discovered jailbreak type, harmful then safe
    harmful_types = discover_jailbreak_types(env, "harmful")
    safe_types = discover_jailbreak_types(env, "safe")

    for jb_type in harmful_types:
        print(f"[{env} | Shard {args.shard_id}] transformed/{jb_type} harmful ...")
        e = run_transformed_split(llm, sampling, tokenizer, env, "harmful", jb_type, base_raw=base_harmful_raw, keep_indices=my_harmful_indices, debug=args.debug)
        write_json(out_dir / "transformed" / jb_type / f"harmful_{args.shard_id}.json", e)

    for jb_type in safe_types:
        print(f"[{env} | Shard {args.shard_id}] transformed/{jb_type} safe ...")
        e = run_transformed_split(llm, sampling, tokenizer, env, "safe", jb_type, base_raw=base_safe_raw, keep_indices=my_safe_indices, debug=args.debug)
        write_json(out_dir / "transformed" / jb_type / f"safe_{args.shard_id}.json", e)

    print(f"[{env} | Shard {args.shard_id}] Done.")


if __name__ == "__main__":
    main()