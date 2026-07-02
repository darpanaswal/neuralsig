"""
prep_behaviors.py — build the GCG training behavior set from your own
environment harmful sets, with train/eval contamination guarded.

The universal GCG suffix is optimized against these behaviors. To keep results
valid, the training behaviors MUST be held out from whatever the probe is
evaluated on. This script supports two holdout strategies:

  --strategy split   : sample N behaviors ACROSS the chosen environments and
                       record their indices so you can exclude them at eval.
  --strategy heldout : draw all N behaviors from a single held-out environment
                       (e.g. the env your run already excludes), so they are
                       excluded from that run's eval by construction.

Outputs:
  behaviors_train.json   : flat JSON array of N behavior strings (for gcg_generate.py)
  behaviors_train_meta.json : provenance — {env: [excluded_indices]} so the eval
                              loader can drop exactly these prompts.

Sampling spans environments by default (a few from each) because the suffix is
applied across all five envs; a cross-env training draw yields a more genuinely
universal suffix than drawing all N from one env.

Usage:
  # cross-env split holdout (recommended for "apply everywhere" pools)
  python prep_behaviors.py --strategy split \
      --envs cybersecurity medical hate_harassment general_crime bioterrorism \
      --n 25 --seed 0

  # single held-out env (matches a leave-one-env-out run)
  python prep_behaviors.py --strategy heldout --heldout_env cybersecurity --n 25
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

ENV_FILE = "{env}_harmful.json"


def _load_env(env: str, data_dir: Path) -> List[str]:
    path = data_dir / ENV_FILE.format(env=env)
    prompts = json.loads(path.read_text())
    if not isinstance(prompts, list):
        raise ValueError(f"{path} is not a flat JSON array")
    return prompts


def strategy_split(
    envs: List[str], n: int, data_dir: Path, rng: random.Random
    ) -> (List[str], Dict[str, List[int]]): # type: ignore
    """Sample n behaviors spread across envs; record excluded indices per env."""
    # Distribute n as evenly as possible across the envs.
    per = [n // len(envs)] * len(envs)
    for i in range(n % len(envs)):
        per[i] += 1

    train: List[str] = []
    excluded: Dict[str, List[int]] = {}
    for env, k in zip(envs, per):
        prompts = _load_env(env, data_dir)
        idxs = sorted(rng.sample(range(len(prompts)), min(k, len(prompts))))
        excluded[env] = idxs
        train.extend(prompts[i] for i in idxs)
    rng.shuffle(train)
    return train, excluded


def strategy_heldout(
    heldout_env: str, n: int, data_dir: Path, rng: random.Random
) -> (List[str], Dict[str, List[int]]): # type: ignore
    """Draw all n behaviors from a single held-out env."""
    prompts = _load_env(heldout_env, data_dir)
    idxs = sorted(rng.sample(range(len(prompts)), min(n, len(prompts))))
    train = [prompts[i] for i in idxs]
    # Entire env is held out for that run; record indices anyway for provenance.
    return train, {heldout_env: idxs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["split", "heldout"], default="split")
    parser.add_argument("--envs", nargs="+",
                        default=["data/cybersecurity", "data/medical", "data/hate_harassment",
                                 "data/general_crime", "data/bioterrorism"],
                        help="envs to sample from (split strategy)")
    parser.add_argument("--heldout_env", default="cybersecurity",
                        help="single env to draw from (heldout strategy)")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--out", default="data/behaviors_train.json")
    parser.add_argument("--meta_out", default="data/behaviors_train_meta.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)

    if args.strategy == "split":
        train, excluded = strategy_split(args.envs, args.n, data_dir, rng)
    else:
        train, excluded = strategy_heldout(args.heldout_env, args.n, data_dir, rng)

    Path(args.out).write_text(json.dumps(train, indent=2, ensure_ascii=False))
    Path(args.meta_out).write_text(json.dumps(
        {"strategy": args.strategy, "n": len(train),
         "seed": args.seed, "excluded_indices": excluded},
        indent=2, ensure_ascii=False))

    print(f"Wrote {len(train)} training behaviors -> {args.out}")
    print(f"Excluded indices (drop these from eval) -> {args.meta_out}")
    for env, idxs in excluded.items():
        print(f"  {env}: {len(idxs)} held out")


if __name__ == "__main__":
    main()