#!/usr/bin/env python3
"""
Offline evaluation of RL (GRPO/PPO) checkpoints for plasticity / forgetting
analysis, mirroring `eval_checkpoints.py` but for verl-style RL runs.

Differences from `eval_checkpoints.py`:

- Inputs are HuggingFace-format RL checkpoints in a directory named
  `.../checkpoints_hf_format/global_step_{N}/` (as produced by the FSDP→HF
  conversion in this repo).
- The "training data" history is recovered from the per-step rollout dumps at
  `.../rollouts/training/{k}.jsonl` (the batch generated at step k, scored,
  then used to update the policy).
- Each checkpoint is evaluated on three scopes:
      in_batch:  rollouts at this very step (file `{N}.jsonl`)
      old_data:  rollouts from earlier stride-eligible steps (k < N)
      new_data:  rollouts from later stride-eligible steps  (k > N)
  Within each scope, samples are split by score into:
      positive = {score == 1.0}
      negative = {score == 0.0}
  Every metric (loss, token entropy, KL-from-init, KL-from-previous-checkpoint,
  dead-units) is reported separately for each (scope, polarity) subset.

Per-checkpoint results are saved to
    {checkpoint_dir}/global_step_{N}/eval_metrics.json
with the structure
    {
      "global_step": N,
      ...
      "in_batch":  { "positive": {...}, "negative": {...}, ... },
      "old_data":  { "positive": {...}, "negative": {...}, ... },
      "new_data":  { "positive": {...}, "negative": {...}, ... },
    }

Single-GPU. KL is computed on-the-fly by running init / prev / cur models on
the same batch. The init model is kept loaded throughout; prev rotates from
the previous checkpoint in the iteration.

Usage:
    python eval_rl_checkpoints.py \
        --checkpoint_dir /home/t-jinshen/amlt/qwen3b_cd_noformat/qwen3b_cd_noformat/qwen3b_cd_noformat/checkpoints_hf_format \
        --init_model_path Qwen/Qwen2.5-3B \
        --step_interval 50

Disable expensive parts via --skip_kl, --skip_dead_units, --skip_in_batch,
--skip_old_data, --skip_new_data.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Reuse helpers from eval_checkpoints.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoints import (  # noqa: E402
    evaluate_dataset,
    free_model,
    load_model,
)
from run_sft import collate_fn  # noqa: E402


# ----------------------------- args ---------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Offline RL-checkpoint eval with pos/neg split")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Directory containing global_step_{N} subdirectories "
                        "(e.g. .../checkpoints_hf_format).")
    p.add_argument("--rollouts_dir", type=str, default=None,
                   help="Directory of {step}.jsonl rollouts. Defaults to "
                        "<run_root>/rollouts/training where <run_root> is the "
                        "parent of --checkpoint_dir.")
    p.add_argument("--init_model_path", type=str, required=True,
                   help="Path/HF id of the initialization (base) model used to start RL.")

    # What to evaluate
    p.add_argument("--step_interval", type=int, default=50,
                   help="Only evaluate global_step_{N} where N is a multiple of this.")
    p.add_argument("--checkpoints", type=str, default=None,
                   help="Comma-separated explicit list of step numbers; overrides --step_interval.")
    p.add_argument("--history_step_stride", type=int, default=0,
                   help="Stride for old/new-data aggregation: include rollouts {k}.jsonl only "
                        "where k is a multiple of this. Defaults to --step_interval.")
    p.add_argument("--skip_kl", action="store_true",
                   help="Skip KL-from-init and KL-from-prev-ckpt.")
    p.add_argument("--skip_dead_units", action="store_true")
    p.add_argument("--skip_in_batch", action="store_true",
                   help="Skip eval on rollouts collected at this very checkpoint step.")
    p.add_argument("--skip_old_data", action="store_true",
                   help="Skip eval on rollouts from earlier stride-eligible steps (k < N).")
    p.add_argument("--skip_new_data", action="store_true",
                   help="Skip eval on rollouts from later stride-eligible steps (k > N).")
    p.add_argument("--positive_score", type=float, default=1.0,
                   help="Records with score == this value form the 'positive' subset.")
    p.add_argument("--negative_score", type=float, default=0.0,
                   help="Records with score == this value form the 'negative' subset.")
    p.add_argument("--max_samples_per_class", type=int, default=256,
                   help="If >0, cap each of positive/negative subsets to this many samples "
                        "(deterministic prefix order). Applied per scope (in_batch / old_data / new_data).")
    p.add_argument("--append_eos", action=argparse.BooleanOptionalAction, default=True,
                   help="Append tokenizer.eos_token_id to each response.")

    # Dead units
    p.add_argument("--dead_unit_epsilon", type=float, default=1e-3)
    p.add_argument("--dead_unit_threshold", type=float, default=0.0,
                   help="A unit is 'dead' if its activation_frequency <= threshold.")

    # Compute
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_seq_length", type=int, default=3072)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    # multi-GPU parallel eval (same slicing contract as eval_checkpoints.py)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=4)
    return p.parse_args()


# ----------------------------- helpers ------------------------------------

def list_hf_checkpoints(ckpt_root: str, step_interval: int,
                        only_steps: list[int] | None) -> list[tuple[int, str]]:
    out = []
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"--checkpoint_dir does not exist: {ckpt_root}")
    for d in os.listdir(ckpt_root):
        m = re.match(r"global_step_(\d+)$", d)
        if not m:
            continue
        step = int(m.group(1))
        full = os.path.join(ckpt_root, d)
        if not os.path.isdir(full):
            continue
        out.append((step, full))
    out.sort(key=lambda x: x[0])
    if only_steps is not None:
        keep = set(only_steps)
        out = [x for x in out if x[0] in keep]
    elif step_interval > 0:
        out = [x for x in out if x[0] % step_interval == 0]
    return out


def default_rollouts_dir(ckpt_dir: str) -> str:
    """If `ckpt_dir` ends in '.../checkpoints_hf_format', sibling 'rollouts/training'."""
    parent = os.path.dirname(os.path.abspath(ckpt_dir))
    candidate = os.path.join(parent, "rollouts", "training")
    return candidate


def load_rollouts(rollouts_dir: str, step: int) -> list[dict]:
    path = os.path.join(rollouts_dir, f"{step}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing rollout file: {path}")
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_line_idx"] = i
            rec["_step"] = step
            out.append(rec)
    return out


def list_rollout_steps(rollouts_dir: str) -> list[int]:
    """Return all integer steps for which `{step}.jsonl` exists, sorted ascending."""
    out = []
    for fname in os.listdir(rollouts_dir):
        m = re.match(r"(\d+)\.jsonl$", fname)
        if m:
            out.append(int(m.group(1)))
    out.sort()
    return out


def load_rollouts_aggregated(rollouts_dir: str, steps: list[int]) -> list[dict]:
    """Concatenate rollouts from a list of step files, skipping any that are missing."""
    out = []
    for k in steps:
        path = os.path.join(rollouts_dir, f"{k}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_line_idx"] = i
                rec["_step"] = k
                out.append(rec)
    return out


# --------------------------- dataset --------------------------------------

class RolloutDataset(Dataset):
    """Wraps rollout records (prompt = `input`, response = `output`) as items
    in the same shape produced by `SFTDataset`, so `collate_fn` works.

    Records whose tokenized full length exceeds `max_seq_length` are dropped.
    """

    def __init__(self, records: list[dict], tokenizer, max_seq_length: int,
                 append_eos: bool, tag: str):
        self.max_seq_length = max_seq_length
        # `eos_ids` is also consumed by gradient_diagnostics in eval_checkpoints.py;
        # keeping the attribute name matches SFTDataset.
        if append_eos and tokenizer.eos_token_id is not None:
            self.eos_ids = [int(tokenizer.eos_token_id)]
        else:
            self.eos_ids = []

        prompt_texts = [r["input"] for r in records]
        response_texts = [r["output"] for r in records]

        prompt_enc = tokenizer(
            prompt_texts, add_special_tokens=False, return_attention_mask=False,
        )
        response_enc = tokenizer(
            response_texts, add_special_tokens=False, return_attention_mask=False,
        )

        self.items = []
        self.skipped_long = 0
        for i, rec in enumerate(records):
            p_ids = prompt_enc["input_ids"][i]
            r_ids = response_enc["input_ids"][i]
            full_len = len(p_ids) + len(r_ids) + len(self.eos_ids)
            if full_len > max_seq_length:
                self.skipped_long += 1
                continue
            self.items.append({
                "prompt_ids": p_ids,
                "response_ids": r_ids,
                "sample_id": f"{tag}@step{rec.get('_step', 0)}:{rec.get('_line_idx', i)}",
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        full_ids = torch.tensor(it["prompt_ids"] + it["response_ids"] + self.eos_ids,
                                dtype=torch.long)
        labels = full_ids.clone()
        labels[: len(it["prompt_ids"])] = -100
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(full_ids),
            "dataset_idx": idx,
            "sample_id": it["sample_id"],
        }


def split_pos_neg(records: list[dict], pos_score: float, neg_score: float,
                  cap: int) -> tuple[list[dict], list[dict]]:
    pos = [r for r in records if float(r.get("score", 0.0)) == pos_score]
    neg = [r for r in records if float(r.get("score", 0.0)) == neg_score]
    if cap and cap > 0:
        pos = pos[:cap]
        neg = neg[:cap]
    return pos, neg


# ----------------------------- main ---------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    rollouts_dir = args.rollouts_dir or default_rollouts_dir(args.checkpoint_dir)
    if not os.path.isdir(rollouts_dir):
        raise FileNotFoundError(f"Rollouts dir not found: {rollouts_dir}")
    print(f"Rollouts dir: {rollouts_dir}")

    # ---------- tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.init_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---------- checkpoints ----------
    only = None
    if args.checkpoints:
        only = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    ckpts = list_hf_checkpoints(args.checkpoint_dir, args.step_interval, only)
    if not ckpts:
        raise RuntimeError(
            f"No global_step_{{N}} subdirs found in {args.checkpoint_dir} "
            f"matching step_interval={args.step_interval} / checkpoints={only}"
        )
    print(f"Will evaluate {len(ckpts)} checkpoints: {[s for s, _ in ckpts]}")

    # ---------- history aggregation steps ----------
    history_stride = args.history_step_stride if args.history_step_stride > 0 else args.step_interval
    need_history = (not args.skip_old_data) or (not args.skip_new_data)
    available_rollout_steps: list[int] = []
    history_steps: list[int] = []
    if need_history:
        available_rollout_steps = list_rollout_steps(rollouts_dir)
        if history_stride > 0:
            history_steps = [k for k in available_rollout_steps if k % history_stride == 0]
        else:
            history_steps = list(available_rollout_steps)
        print(f"  history step stride = {history_stride}; "
              f"{len(history_steps)} eligible rollout files for old/new aggregation")

    # ---------- multi-GPU parallel slicing ----------
    predecessor_ckpt = None
    if args.world_size > 1:
        n = len(ckpts)
        lo = args.rank * n // args.world_size
        hi = (args.rank + 1) * n // args.world_size
        if lo > 0:
            predecessor_ckpt = ckpts[lo - 1]
        ckpts = ckpts[lo:hi]
        print(f"  [rank {args.rank}/{args.world_size}] my slice: steps {[s for s, _ in ckpts]}"
              + (f" (predecessor for KL: step {predecessor_ckpt[0]})" if predecessor_ckpt else ""))
        if not ckpts:
            print("  Empty slice for this rank; nothing to do.")
            return

    # ---------- init model (kept loaded for KL-from-init) ----------
    init_model = None
    if not args.skip_kl:
        print(f"Loading init model from {args.init_model_path} ...")
        init_model = load_model(args.init_model_path, dtype, device)

    # ---------- iterate checkpoints ----------
    prev_model = None
    prev_step = None
    if predecessor_ckpt is not None and not args.skip_kl:
        pstep, pdir = predecessor_ckpt
        print(f"  [rank {args.rank}] preloading predecessor step {pstep} as prev_model ...")
        prev_model = load_model(pdir, dtype, device)
        prev_step = pstep

    for step, ckpt_dir in ckpts:
        out_path = os.path.join(ckpt_dir, "eval_metrics.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[step {step}] eval_metrics.json exists, skipping (use --overwrite to redo)")
            free_model(prev_model)
            prev_model = None
            if not args.skip_kl:
                print(f"  Loading {ckpt_dir} as next prev_model ...")
                prev_model = load_model(ckpt_dir, dtype, device)
            prev_step = step
            continue

        print(f"\n=== Checkpoint step {step}: {ckpt_dir} ===")

        # ---- gather records per scope ----
        scopes: list[tuple[str, list[dict], list[int]]] = []  # (name, records, source_steps)
        if not args.skip_in_batch:
            try:
                in_batch_records = load_rollouts(rollouts_dir, step)
                scopes.append(("in_batch", in_batch_records, [step]))
            except FileNotFoundError as e:
                print(f"  WARNING: in_batch: {e}")
        if not args.skip_old_data:
            old_steps = [k for k in history_steps if k < step]
            old_records = load_rollouts_aggregated(rollouts_dir, old_steps)
            scopes.append(("old_data", old_records, old_steps))
        if not args.skip_new_data:
            new_steps = [k for k in history_steps if k > step]
            new_records = load_rollouts_aggregated(rollouts_dir, new_steps)
            scopes.append(("new_data", new_records, new_steps))

        if not scopes:
            print("  All scopes skipped; nothing to evaluate at this checkpoint.")
            continue

        # ---- load current model ----
        cur_model = load_model(ckpt_dir, dtype, device)

        result = {
            "global_step": step,
            "ckpt_dir": ckpt_dir,
            "rollouts_dir": rollouts_dir,
            "init_model_path": args.init_model_path,
            "prev_step": prev_step,
            "positive_score": args.positive_score,
            "negative_score": args.negative_score,
            "history_step_stride": history_stride,
            "max_samples_per_class": args.max_samples_per_class,
        }

        for scope_name, recs, source_steps in scopes:
            pos_recs, neg_recs = split_pos_neg(
                recs, args.positive_score, args.negative_score, args.max_samples_per_class,
            )
            print(f"  [{scope_name}] source_steps={len(source_steps)} "
                  f"total={len(recs)} "
                  f"pos(score=={args.positive_score})={len(pos_recs)} "
                  f"neg(score=={args.negative_score})={len(neg_recs)}")

            scope_block = {
                "num_records": len(recs),
                "source_steps": source_steps,
                "num_source_steps": len(source_steps),
                "num_positives_pre_cap": sum(
                    1 for r in recs if float(r.get("score", 0.0)) == args.positive_score
                ),
                "num_negatives_pre_cap": sum(
                    1 for r in recs if float(r.get("score", 0.0)) == args.negative_score
                ),
            }

            for tag, sub_recs in [("positive", pos_recs), ("negative", neg_recs)]:
                if not sub_recs:
                    scope_block[tag] = {"size": 0}
                    print(f"    {tag}: no samples")
                    continue
                ds = RolloutDataset(
                    sub_recs, tokenizer,
                    max_seq_length=args.max_seq_length,
                    append_eos=args.append_eos,
                    tag=f"{scope_name}/{tag}",
                )
                if len(ds) == 0:
                    print(f"    {tag}: all {len(sub_recs)} samples exceeded max_seq_length, skipping")
                    scope_block[tag] = {
                        "size": 0,
                        "size_pre_filter": len(sub_recs),
                        "skipped_long": ds.skipped_long,
                    }
                    continue
                loader = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                metrics = evaluate_dataset(
                    cur_model, init_model, prev_model, loader, device,
                    do_dead_units=not args.skip_dead_units,
                    epsilon=args.dead_unit_epsilon,
                    dead_threshold=args.dead_unit_threshold,
                    desc=f"{scope_name}/{tag}@{step}",
                )
                metrics["size"] = len(ds)
                metrics["size_pre_filter"] = len(sub_recs)
                metrics["skipped_long"] = ds.skipped_long
                scope_block[tag] = metrics
                print(f"    {tag}: n={len(ds)} loss={metrics['loss']:.4f} "
                      f"ent={metrics['token_entropy']:.3f}"
                      + (f" kl_init={metrics.get('kl_from_init', float('nan')):.4e}"
                         if "kl_from_init" in metrics else "")
                      + (f" kl_to_init={metrics.get('kl_to_init', float('nan')):.4e}"
                         if "kl_to_init" in metrics else "")
                      + (f" kl_prev={metrics.get('kl_from_previous_checkpoint', float('nan')):.4e}"
                         if "kl_from_previous_checkpoint" in metrics else "")
                      + (f" kl_to_prev={metrics.get('kl_to_previous_checkpoint', float('nan')):.4e}"
                         if "kl_to_previous_checkpoint" in metrics else "")
                      )

            result[scope_name] = scope_block

        # ---- save ----
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> {out_path}")

        # ---- rotate prev ----
        free_model(prev_model)
        prev_model = cur_model
        prev_step = step

    # ---- cleanup ----
    free_model(prev_model)
    free_model(init_model)
    print("\nDone.")


if __name__ == "__main__":
    main()
