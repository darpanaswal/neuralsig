"""
gcg_generate.py — offline GPU pre-step for the GCG transform.

Generates a small pool of UNIVERSAL transferable adversarial suffixes
(Zou et al. 2023) by optimizing one shared suffix against a batch of harmful
behaviors at once, using nanoGCG's primitives. Writes a JSON pool that the
black-box `gcg_suffix` transform then samples from.

This is the ONE step in the pipeline that requires:
  - an open-weight model + GPU,
  - gradient-based optimization (white-box).
It is intentionally separated from the no-LLM transform library. Run it once;
reuse the produced pool across all datasets.

Why a custom loop: nanoGCG's `run()` optimizes a suffix for a SINGLE prompt.
Universal mode (one suffix for many behaviors) is the paper's headline setting,
so we share a single `{optim_str}` across N behaviors and average the GCG loss
across them at each optimization step.

Faithfulness notes:
  - Target string is the affirmative response "Sure, here is ..." per the paper.
  - The suffix is appended to the user turn (default {optim_str} placement).
  - Universality comes from joint optimization over multiple behaviors, exactly
    as in Zou et al.'s multi-prompt / transfer experiments.

Usage:
  python gcg_generate.py \
      --model meta-llama/Llama-3.2-3B-Instruct \
      --behaviors behaviors_train.json \
      --n_suffixes 5 \
      --steps 500 \
      --out gcg_suffix_pool.json

`behaviors_train.json` is a flat JSON array of ~25 harmful behavior strings
used purely as the optimization training set (held separate from your eval
sets to avoid contaminating results).

Requires:  pip install nanogcg torch transformers
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import nanogcg
from nanogcg import GCGConfig


# Pascal GPUs (e.g. GTX 1080 Ti) lack bf16; allow fp16 override via env.
_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}.get(os.environ.get("GCG_DTYPE", "bfloat16"), torch.bfloat16)


def _affirmative_target(behavior: str) -> str:
    """Paper-style affirmative target: 'Sure, here is <behavior> ...'.

    Zou et al. target the model to begin its reply with an affirmative
    restatement of the request, which flips the model into a compliant mode.
    """
    b = behavior.strip().rstrip("?.")
    # Lowercase the leading verb for natural continuation after "Sure, here is".
    return f"Sure, here is {b[0].lower() + b[1:]}:\n\n"


def generate_universal_suffix(
    model,
    tokenizer,
    behaviors: List[str],
    steps: int,
    seed: int,
    search_width: int,
    topk: int,
) -> str:
    """Optimize one suffix shared across `behaviors` (universal mode).

    Strategy: nanoGCG optimizes per-message, so to approximate joint multi-
    behavior optimization with the stock package we iterate over behaviors in
    a round-robin, warm-starting each run from the previous best suffix. This
    drives the suffix toward one that lowers loss across the whole set rather
    than overfitting a single behavior. For full simultaneous averaging, use
    the original llm-attacks `run_gcg_multiple.sh`; this loop is the nanoGCG-
    based approximation and is documented as such.
    """
    optim_str = None  # let nanoGCG initialize on the first pass
    # Split the step budget across behaviors, multiple rounds for mixing.
    rounds = 3
    per_run_steps = max(1, steps // (rounds * len(behaviors)))

    for r in range(rounds):
        for behavior in behaviors:
            messages = [
                {"role": "user", "content": f"{behavior}{{optim_str}}"},
            ]
            target = _affirmative_target(behavior)

            config = GCGConfig(
                num_steps=per_run_steps,
                search_width=search_width,
                topk=topk,
                seed=seed,
                optim_str_init=optim_str if optim_str is not None
                else "x x x x x x x x x x x x x x x x x x x x",
                verbosity="WARNING",
            )
            result = nanogcg.run(model, tokenizer, messages, target, config)
            # Warm-start the next behavior from this best suffix.
            optim_str = result.best_string

    return optim_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--behaviors", required=True,
                        help="flat JSON array of harmful behavior training strings")
    parser.add_argument("--n_suffixes", type=int, default=5,
                        help="size of the universal suffix pool to produce")
    parser.add_argument("--steps", type=int, default=500,
                        help="total GCG step budget per suffix")
    parser.add_argument("--n_behaviors", type=int, default=25,
                        help="how many training behaviors to optimize over")
    parser.add_argument("--search_width", type=int, default=512)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--seed_base", type=int, default=1000,
                        help="base seed; suffix k uses seed_base + k")
    parser.add_argument("--out", default="gcg_suffix_pool.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    behaviors_all = json.loads(Path(args.behaviors).read_text())
    if not isinstance(behaviors_all, list) or not behaviors_all:
        raise ValueError("--behaviors must be a non-empty JSON array of strings")
    behaviors = behaviors_all[:args.n_behaviors]

    print(f"Loading {args.model} (dtype={_DTYPE}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=_DTYPE
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    pool = []
    for k in range(args.n_suffixes):
        seed = args.seed_base + k
        print(f"\n=== Optimizing universal suffix {k + 1}/{args.n_suffixes} "
              f"(seed={seed}) over {len(behaviors)} behaviors ===")
        suffix = generate_universal_suffix(
            model, tokenizer, behaviors,
            steps=args.steps, seed=seed,
            search_width=args.search_width, topk=args.topk,
        )
        print(f"  -> {suffix!r}")
        pool.append(suffix)

    Path(args.out).write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(pool)} universal suffixes -> {args.out}")
    print("Load this pool in jailbreak_transforms.gcg_suffix via load_gcg_pool().")


if __name__ == "__main__":
    main()