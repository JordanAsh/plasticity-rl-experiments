#!/usr/bin/env python3
"""
Offline evaluation of SFT checkpoints for plasticity / forgetting analysis.

Iterates over checkpoints saved by run_sft.py and, for each, computes:

  Probe set (KK eval / parquet, used as fixed probe — never trained on):
    - probe_loss
    - probe_token_entropy
    - probe_kl_from_init
    - probe_kl_from_previous_checkpoint
    - probe_dead_units
    - (optional) probe_gradient_norm / variance / noise_scale

  Old data (samples actually trained before this checkpoint, per manifests):
    - old_data_loss
    - old_data_token_entropy
    - old_data_kl_from_init
    - old_data_kl_from_previous_checkpoint
    - old_data_dead_units
    - (optional) old_data_gradient_norm / variance / noise_scale

Per-checkpoint results are saved to {ckpt}/eval_metrics.json.

Single-GPU. KL is computed on-the-fly by running init / prev / cur models on the
same batch (init kept loaded throughout; prev rotates from the previous ckpt).

Usage:
    python eval_checkpoints.py \
        --run_dir sft_outputs/seed42_shuffled \
        --probe_parquet ~/data/kk/test.parquet \
        --probe_max_samples 256

Disable expensive parts via --skip_old_data, --skip_kl, --skip_dead_units.
Enable gradient diagnostics via --grad_microbatches 8 (default 0 = off).
"""

import argparse
import gc
import json
import os
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Reuse helpers from run_sft.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sft import (  # noqa: E402
    SFTDataset,
    collate_fn,
    load_positives,
)


# ----------------------------- args ---------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Offline checkpoint eval for SFT plasticity")
    p.add_argument("--run_dir", type=str, required=True,
                   help="SFT output directory (contains checkpoints/, manifests/, sft_config.json)")
    p.add_argument("--init_model_path", type=str, default=None,
                   help="Path/HF id of the initialization model. Defaults to model_path from sft_config.json.")
    p.add_argument("--generation_logs_dir", type=str, default=None,
                   help="Override generation_logs_dir for old-data reconstruction. "
                        "Defaults to value from sft_config.json.")

    # Probe set
    p.add_argument("--probe_parquet", type=str,
                   default=os.path.expanduser("~/data/kk/test.parquet"),
                   help="Parquet with KK eval prompts; ground_truth used as completion for loss.")
    p.add_argument("--probe_max_samples", type=int, default=256,
                   help="Cap on number of probe examples (deterministic prefix).")

    # What to evaluate
    p.add_argument("--skip_probe", action="store_true")
    p.add_argument("--skip_old_data", action="store_true")
    p.add_argument("--old_data_step_stride", type=int, default=0,
                   help="If > 0, only include old-data samples from training steps that are multiples "
                        "of this stride (e.g. 50 or 100). Keeps later-checkpoint old-data eval bounded "
                        "and gives every checkpoint a uniformly-spaced slice of its own history.")
    p.add_argument("--skip_kl", action="store_true",
                   help="Skip KL-from-init and KL-from-prev-ckpt.")
    p.add_argument("--skip_dead_units", action="store_true")
    p.add_argument("--checkpoints", type=str, default=None,
                   help="Comma-separated list of checkpoint step numbers; default = all found.")
    p.add_argument("--include_final", action="store_true",
                   help="Also evaluate {run_dir}/final as the last checkpoint.")

    # Gradient diagnostics
    p.add_argument("--grad_microbatches", type=int, default=0,
                   help="If >0, compute mean grad norm / variance / noise scale via M microbatches.")
    p.add_argument("--grad_max_samples", type=int, default=2048,
                   help="Cap on samples used for gradient diagnostics (per dataset).")
    p.add_argument("--grad_on_probe", action="store_true",
                   help="Also compute gradient diagnostics on probe set (default: only on old data).")

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
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if eval_metrics.json already exists for a checkpoint.")
    # multi-GPU parallel eval: contiguously slice the checkpoint list across ranks
    p.add_argument("--rank", type=int, default=0,
                   help="This worker's index when running multi-GPU parallel eval (0..world_size-1).")
    p.add_argument("--world_size", type=int, default=1,
                   help="Total number of parallel eval workers (default 1 = no slicing).")
    return p.parse_args()


# ----------------------------- helpers ------------------------------------

def load_run_config(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "sft_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


def list_checkpoints(run_dir: str, only_steps: list[int] | None) -> list[tuple[int, str]]:
    out = []
    base = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(base):
        for d in os.listdir(base):
            m = re.match(r"step_(\d+)$", d)
            if m:
                out.append((int(m.group(1)), os.path.join(base, d)))
    out.sort(key=lambda x: x[0])
    if only_steps is not None:
        keep = set(only_steps)
        out = [x for x in out if x[0] in keep]
    return out


def merge_manifests(run_dir: str) -> list[dict]:
    """Return all manifest entries across all ranks, sorted by global_step."""
    files = sorted(glob(os.path.join(run_dir, "manifests", "rank_*.jsonl")))
    entries = []
    for mf in files:
        with open(mf) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    entries.sort(key=lambda e: (e["global_step"], e.get("rank", 0)))
    return entries


def cumulative_seen_per_checkpoint(entries: list[dict], ckpt_steps: list[int],
                                   step_stride: int = 0) -> dict[int, set]:
    """For each checkpoint step, return the set of sample_ids seen up to and including it.

    If step_stride > 0, only entries whose global_step is a multiple of step_stride contribute,
    yielding a deterministic uniformly-spaced subsample of training history.
    """
    out: dict[int, set] = {}
    ckpt_steps_sorted = sorted(ckpt_steps)
    ci = 0
    running: set = set()
    for e in entries:
        if step_stride <= 0 or (e["global_step"] % step_stride == 0):
            for mb in e["micro_batches"]:
                running.update(mb["sample_ids"])
        while ci < len(ckpt_steps_sorted) and e["global_step"] >= ckpt_steps_sorted[ci]:
            out[ckpt_steps_sorted[ci]] = set(running)
            ci += 1
        if ci >= len(ckpt_steps_sorted):
            break
    # Any remaining (checkpoints beyond what we've seen) get the final running set.
    for cs in ckpt_steps_sorted[ci:]:
        out[cs] = set(running)
    return out


# --------------------------- probe dataset --------------------------------

class KKProbeDataset(Dataset):
    """Fixed probe set: KK eval parquet, with ground_truth used as the completion.

    Each item carries `sample_id = "probe:{i}"`. Items longer than max_seq_length
    are dropped.
    """

    def __init__(self, parquet_path: str, tokenizer, max_seq_length: int, max_samples: int | None):
        df = pd.read_parquet(parquet_path)
        if max_samples is not None and len(df) > max_samples:
            df = df.iloc[:max_samples].reset_index(drop=True)

        self.eos_ids = tokenizer("<|im_end|>", add_special_tokens=False).input_ids
        self.items = []

        for i, row in df.iterrows():
            messages = list(row["prompt"])
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            gt = row["reward_model"]["ground_truth"]
            if isinstance(gt, dict):
                response_text = json.dumps(gt)
            elif not isinstance(gt, str):
                response_text = str(gt)
            else:
                response_text = gt

            p_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            r_ids = tokenizer(response_text, add_special_tokens=False).input_ids
            full_len = len(p_ids) + len(r_ids) + len(self.eos_ids)
            if full_len > max_seq_length:
                continue
            self.items.append({
                "prompt_ids": p_ids,
                "response_ids": r_ids,
                "sample_id": f"probe:{int(i)}",
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        full_ids = torch.tensor(it["prompt_ids"] + it["response_ids"] + self.eos_ids, dtype=torch.long)
        labels = full_ids.clone()
        labels[: len(it["prompt_ids"])] = -100
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(full_ids),
            "dataset_idx": idx,
            "sample_id": it["sample_id"],
        }


# --------------------------- dead-unit tracker ----------------------------

class DeadUnitTracker:
    """Hooks down_proj of every MLP block and accumulates activation frequencies
    of intermediate units (i.e., values of `silu(gate_proj) * up_proj`) that
    exceed `epsilon` in absolute value. Counts only positions where labels != -100.
    """

    def __init__(self, model, epsilon: float, dead_threshold: float):
        self.epsilon = epsilon
        self.dead_threshold = dead_threshold
        self.layer_active: dict[int, torch.Tensor] = {}
        self.layer_total: dict[int, int] = {}
        self.handles = []
        self.current_mask: torch.Tensor | None = None  # set per-batch externally

        pat = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.down_proj$")
        for name, module in model.named_modules():
            m = pat.search(name)
            if m:
                layer_idx = int(m.group(1))
                self.handles.append(
                    module.register_forward_pre_hook(self._make_hook(layer_idx))
                )

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs):
            if self.current_mask is None:
                return
            x = inputs[0].detach()  # (B, T, intermediate_size)
            mask = self.current_mask  # (B, T) bool
            if x.shape[:2] != mask.shape:
                # If shapes mismatch (e.g. batch processed differently), skip.
                return
            active = (x.abs() > self.epsilon)  # (B, T, I)
            masked_active = (active & mask.unsqueeze(-1)).sum(dim=(0, 1))  # (I,)
            n = int(mask.sum().item())
            if layer_idx not in self.layer_active:
                self.layer_active[layer_idx] = masked_active.to(torch.float64)
                self.layer_total[layer_idx] = n
            else:
                self.layer_active[layer_idx] += masked_active.to(torch.float64)
                self.layer_total[layer_idx] += n

        return hook

    def reset(self):
        self.layer_active = {}
        self.layer_total = {}
        self.current_mask = None

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def summary(self) -> dict:
        out = {}
        all_dead = 0
        all_total = 0
        for layer_idx in sorted(self.layer_active.keys()):
            active = self.layer_active[layer_idx]
            total = self.layer_total[layer_idx]
            if total == 0:
                continue
            freq = (active / total).cpu().numpy()
            dead_mask = freq <= self.dead_threshold
            entry = {
                "intermediate_size": int(freq.shape[0]),
                "dead_count": int(dead_mask.sum()),
                "dead_fraction": float(dead_mask.mean()),
                "mean_activation_freq": float(freq.mean()),
                "min_activation_freq": float(freq.min()),
            }
            out[f"layer_{layer_idx}"] = entry
            all_dead += entry["dead_count"]
            all_total += entry["intermediate_size"]
        if all_total > 0:
            out["_overall"] = {
                "dead_count": all_dead,
                "intermediate_total": all_total,
                "dead_fraction": all_dead / all_total,
            }
        return out


# --------------------------- evaluation core ------------------------------

@torch.no_grad()
def evaluate_dataset(
    cur_model,
    init_model,
    prev_model,
    dataloader,
    device,
    do_dead_units: bool,
    epsilon: float,
    dead_threshold: float,
    desc: str = "eval",
) -> dict:
    cur_model.eval()
    if init_model is not None:
        init_model.eval()
    if prev_model is not None:
        prev_model.eval()

    # Dead-unit tracker on the *current* model (probe of where the network is now).
    tracker = None
    if do_dead_units:
        tracker = DeadUnitTracker(cur_model, epsilon, dead_threshold)

    loss_sum = 0.0
    ent_sum = 0.0
    tok_count = 0
    kl_init_sum = 0.0
    kl_prev_sum = 0.0
    kl_tok_count = 0

    try:
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            shift_labels = labels[:, 1:]
            sup_mask = shift_labels.ne(-100)  # (B, T-1)
            n_sup = int(sup_mask.sum().item())
            if n_sup == 0:
                continue

            if tracker is not None:
                tracker.current_mask = labels.ne(-100)  # (B, T)

            cur_logits = cur_model(input_ids=input_ids, attention_mask=attn_mask).logits
            cur_logp = F.log_softmax(cur_logits[:, :-1, :].float(), dim=-1)
            del cur_logits

            # Loss = mean NLL over supervised tokens.
            true_lp = cur_logp.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            loss_sum += float((-(true_lp) * sup_mask).sum().item())

            # Entropy of current model over supervised positions.
            cur_p = cur_logp.exp()
            entropy = -(cur_p * cur_logp).sum(-1)  # (B, T-1)
            ent_sum += float((entropy * sup_mask).sum().item())
            del cur_p, entropy, true_lp
            tok_count += n_sup

            # KL from init: KL(p_init || p_cur) summed over supervised positions.
            if init_model is not None:
                init_logits = init_model(input_ids=input_ids, attention_mask=attn_mask).logits
                init_logp = F.log_softmax(init_logits[:, :-1, :].float(), dim=-1)
                del init_logits
                init_p = init_logp.exp()
                kl_init = (init_p * (init_logp - cur_logp)).sum(-1)  # (B, T-1)
                kl_init_sum += float((kl_init * sup_mask).sum().item())
                del init_p, kl_init, init_logp

            if prev_model is not None:
                prev_logits = prev_model(input_ids=input_ids, attention_mask=attn_mask).logits
                prev_logp = F.log_softmax(prev_logits[:, :-1, :].float(), dim=-1)
                del prev_logits
                prev_p = prev_logp.exp()
                kl_prev = (prev_p * (prev_logp - cur_logp)).sum(-1)
                kl_prev_sum += float((kl_prev * sup_mask).sum().item())
                del prev_p, kl_prev, prev_logp

            if init_model is not None or prev_model is not None:
                kl_tok_count += n_sup

            del cur_logp
    finally:
        if tracker is not None:
            tracker.remove()

    metrics = {
        "loss": loss_sum / max(tok_count, 1),
        "token_entropy": ent_sum / max(tok_count, 1),
        "num_supervised_tokens": tok_count,
    }
    if init_model is not None:
        metrics["kl_from_init"] = kl_init_sum / max(kl_tok_count, 1)
    if prev_model is not None:
        metrics["kl_from_previous_checkpoint"] = kl_prev_sum / max(kl_tok_count, 1)
    if tracker is not None:
        metrics["dead_units"] = tracker.summary()
    return metrics


# --------------------------- gradient diagnostics --------------------------

def _build_microbatch_groups(items_token_counts: list[tuple[int, int]], M: int) -> list[list[int]]:
    """Greedy partition of (idx, n_tokens) into M groups balancing total tokens."""
    groups: list[list[int]] = [[] for _ in range(M)]
    group_totals = [0] * M
    for idx, n in sorted(items_token_counts, key=lambda x: -x[1]):
        g = group_totals.index(min(group_totals))
        groups[g].append(idx)
        group_totals[g] += n
    return groups


def gradient_diagnostics(
    cur_model,
    dataset,
    pad_token_id: int,
    num_microbatches: int,
    max_samples: int,
    device,
    desc: str = "grad",
) -> dict:
    """Compute mean grad norm / per-param variance / noise scale via M
    token-balanced microbatches. Does NOT call optimizer.step.

    Uses Welford-free single-pass: keeps sum_g and sum||g||^2.
    """
    if num_microbatches <= 0 or len(dataset) == 0:
        return {}

    # Build (idx, token_count) list
    n_take = min(len(dataset), max_samples)
    indices = list(range(n_take))
    token_counts = []
    for i in indices:
        it = dataset.items[i] if hasattr(dataset, "items") else dataset.dataset.items[dataset.indices[i]]
        token_counts.append((i, len(it["prompt_ids"]) + len(it["response_ids"]) + len(dataset.eos_ids if hasattr(dataset, "eos_ids") else dataset.dataset.eos_ids)))

    groups = _build_microbatch_groups(token_counts, num_microbatches)

    cur_model.train()  # need grads (will not step)
    # Disable dropout etc during diagnostics is fine for small models; train() turns it on.
    # For diagnostics we want deterministic-ish behaviour: switch back to eval but enable grad.
    cur_model.eval()

    sum_g: torch.Tensor | None = None  # CPU fp32
    sum_g_sq = 0.0
    M_done = 0

    for g_idx, group in enumerate(tqdm(groups, desc=desc, leave=False)):
        if len(group) == 0:
            continue

        # zero grads
        for p in cur_model.parameters():
            if p.grad is not None:
                p.grad = None

        n_tokens = 0
        # Process examples in mini-chunks to bound memory.
        i = 0
        while i < len(group):
            j = min(i + 4, len(group))  # micro-mini-batch size
            ex = [dataset[k] for k in group[i:j]]
            batch = collate_fn(ex, pad_token_id)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            n_sup = int((labels[:, 1:] != -100).sum().item())
            if n_sup == 0:
                i = j
                continue
            out = cur_model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            # HF loss is mean over supervised tokens; un-normalize so accumulated grads
            # equal sum-over-tokens; we'll divide by total later.
            (out.loss * n_sup).backward()
            n_tokens += n_sup
            del out
            i = j

        if n_tokens == 0:
            continue

        # Normalize accumulated gradient by total supervised tokens in this group.
        with torch.no_grad():
            flats = []
            for p in cur_model.parameters():
                if p.grad is not None:
                    p.grad.div_(n_tokens)
                    flats.append(p.grad.detach().float().flatten().cpu())
                    p.grad = None
            g_flat = torch.cat(flats) if flats else None

        if g_flat is None:
            continue

        sum_g_sq += float(g_flat.pow(2).sum().item())
        if sum_g is None:
            sum_g = g_flat.clone()
        else:
            sum_g.add_(g_flat)
        del g_flat
        M_done += 1

    cur_model.eval()  # restore
    for p in cur_model.parameters():
        if p.grad is not None:
            p.grad = None

    if sum_g is None or M_done == 0:
        return {"num_microbatches": 0}

    mean_g = sum_g / M_done
    mean_norm_sq = float(mean_g.pow(2).sum().item())
    mean_norm = mean_norm_sq ** 0.5
    P = sum_g.numel()
    # trace(cov) = (sum||g||^2 - M ||mean||^2) / (M-1)
    trace_cov = max((sum_g_sq - M_done * mean_norm_sq) / max(M_done - 1, 1), 0.0)
    return {
        "num_microbatches": M_done,
        "mean_gradient_norm": mean_norm,
        "gradient_variance_per_param": trace_cov / P,
        "gradient_noise_scale": trace_cov / max(mean_norm_sq, 1e-30),
        "trace_covariance": trace_cov,
        "num_parameters": int(P),
    }


# --------------------------- model loading helpers ------------------------

def load_model(path: str, dtype: torch.dtype, device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def free_model(model):
    if model is None:
        return
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------- main ---------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # ---------- run config ----------
    cfg = load_run_config(args.run_dir)
    init_model_path = args.init_model_path or cfg["model_path"]
    gen_logs_dir = args.generation_logs_dir or cfg["generation_logs_dir"]

    # ---------- tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(init_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---------- probe dataset ----------
    probe_loader = None
    probe_ds = None
    if not args.skip_probe:
        print(f"Loading probe set from {args.probe_parquet} ...")
        probe_ds = KKProbeDataset(
            args.probe_parquet, tokenizer,
            max_seq_length=args.max_seq_length,
            max_samples=args.probe_max_samples,
        )
        print(f"  Probe size: {len(probe_ds)}")
        probe_loader = DataLoader(
            probe_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # ---------- old-data dataset (full SFT dataset) ----------
    full_sft_ds = None
    seen_per_ckpt: dict[int, set] = {}
    if not args.skip_old_data:
        print(f"Reloading SFT positives from {gen_logs_dir} ...")
        positives = load_positives(
            gen_logs_dir, score_threshold=cfg.get("score_threshold", 0.0)
        )
        # Re-apply the same ordering used at training time. We don't actually need
        # the order for old-data eval (we look up by sample_id), but rebuilding
        # the dataset gives us the prompt_ids/response_ids identical to training.
        full_sft_ds = SFTDataset(positives, tokenizer, args.max_seq_length)
        print(f"  Full SFT dataset (post-filter): {len(full_sft_ds)}")

    # ---------- checkpoints ----------
    only = None
    if args.checkpoints:
        only = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    ckpts = list_checkpoints(args.run_dir, only)
    if args.include_final and only is None:
        final_dir = os.path.join(args.run_dir, "final")
        if os.path.isdir(final_dir):
            last_step = ckpts[-1][0] + 1 if ckpts else 0
            ckpts.append((last_step, final_dir))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {args.run_dir}/checkpoints/")
    print(f"Will evaluate {len(ckpts)} checkpoints: {[s for s, _ in ckpts]}")

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

    if not args.skip_old_data:
        entries = merge_manifests(args.run_dir)
        seen_per_ckpt = cumulative_seen_per_checkpoint(
            entries, [s for s, _ in ckpts], step_stride=args.old_data_step_stride,
        )
        if args.old_data_step_stride > 0:
            sizes = {s: len(seen_per_ckpt.get(s, set())) for s, _ in ckpts}
            print(f"  Old-data step stride = {args.old_data_step_stride}; per-ckpt sizes: {sizes}")

    # ---------- init model (kept loaded for KL-from-init) ----------
    init_model = None
    if not args.skip_kl:
        print(f"Loading init model from {init_model_path} ...")
        init_model = load_model(init_model_path, dtype, device)

    # ---------- iterate checkpoints ----------
    prev_model = None
    prev_step = None
    # If we're a non-leading rank, preload the predecessor checkpoint as prev_model
    # so the KL_prev chain stays correct across rank boundaries.
    if predecessor_ckpt is not None and not args.skip_kl:
        pstep, pdir = predecessor_ckpt
        psub = os.path.join(pdir, "model") if os.path.isdir(os.path.join(pdir, "model")) else pdir
        print(f"  [rank {args.rank}] preloading predecessor step {pstep} as prev_model ...")
        prev_model = load_model(psub, dtype, device)
        prev_step = pstep
    for step, ckpt_dir in ckpts:
        out_path = os.path.join(ckpt_dir, "eval_metrics.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[step {step}] eval_metrics.json exists, skipping (use --overwrite to redo)")
            # still need to set this as prev for next iteration
            free_model(prev_model)
            prev_model = None
            if not args.skip_kl:
                print(f"  Loading {ckpt_dir}/model as next prev_model ...")
                model_subdir = os.path.join(ckpt_dir, "model") if os.path.isdir(os.path.join(ckpt_dir, "model")) else ckpt_dir
                prev_model = load_model(model_subdir, dtype, device)
            prev_step = step
            continue

        print(f"\n=== Checkpoint step {step}: {ckpt_dir} ===")
        model_subdir = os.path.join(ckpt_dir, "model")
        if not os.path.isdir(model_subdir):
            # 'final' dir saves directly into ckpt_dir
            model_subdir = ckpt_dir
        cur_model = load_model(model_subdir, dtype, device)

        result = {
            "global_step": step,
            "ckpt_dir": ckpt_dir,
            "init_model_path": init_model_path,
            "prev_step": prev_step,
        }

        # ---- probe ----
        if probe_loader is not None:
            probe_metrics = evaluate_dataset(
                cur_model, init_model, prev_model, probe_loader, device,
                do_dead_units=not args.skip_dead_units,
                epsilon=args.dead_unit_epsilon,
                dead_threshold=args.dead_unit_threshold,
                desc=f"probe@{step}",
            )
            if args.grad_microbatches > 0 and args.grad_on_probe:
                probe_metrics["gradient"] = gradient_diagnostics(
                    cur_model, probe_ds, tokenizer.pad_token_id,
                    args.grad_microbatches, args.grad_max_samples, device,
                    desc=f"probe-grad@{step}",
                )
            result["probe"] = probe_metrics
            print(f"  probe: loss={probe_metrics['loss']:.4f} ent={probe_metrics['token_entropy']:.3f}"
                  + (f" kl_init={probe_metrics.get('kl_from_init', float('nan')):.4e}" if "kl_from_init" in probe_metrics else "")
                  + (f" kl_prev={probe_metrics.get('kl_from_previous_checkpoint', float('nan')):.4e}" if "kl_from_previous_checkpoint" in probe_metrics else "")
                  )

        # ---- old data ----
        if full_sft_ds is not None and not args.skip_old_data:
            seen = seen_per_ckpt.get(step, set())
            old_indices = [i for i, it in enumerate(full_sft_ds.items) if it["sample_id"] in seen]
            if args.old_data_step_stride > 0:
                print(f"  old data size = {len(old_indices)} (stride={args.old_data_step_stride} @ step {step})")
            else:
                print(f"  old data size = {len(old_indices)} (cumulative @ step {step})")
            if old_indices:
                old_subset = Subset(full_sft_ds, old_indices)
                # Subset doesn't expose .items / .eos_ids; expose via attributes for grad helper.
                old_subset.items = [full_sft_ds.items[i] for i in old_indices]
                old_subset.eos_ids = full_sft_ds.eos_ids
                old_loader = DataLoader(
                    old_subset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                old_metrics = evaluate_dataset(
                    cur_model, init_model, prev_model, old_loader, device,
                    do_dead_units=not args.skip_dead_units,
                    epsilon=args.dead_unit_epsilon,
                    dead_threshold=args.dead_unit_threshold,
                    desc=f"old@{step}",
                )
                if args.grad_microbatches > 0:
                    old_metrics["gradient"] = gradient_diagnostics(
                        cur_model, old_subset, tokenizer.pad_token_id,
                        args.grad_microbatches, args.grad_max_samples, device,
                        desc=f"old-grad@{step}",
                    )
                result["old_data"] = old_metrics
                result["old_data_size"] = len(old_indices)
                result["old_data_step_stride"] = args.old_data_step_stride
                print(f"  old:   loss={old_metrics['loss']:.4f} ent={old_metrics['token_entropy']:.3f}"
                      + (f" kl_init={old_metrics.get('kl_from_init', float('nan')):.4e}" if "kl_from_init" in old_metrics else "")
                      + (f" kl_prev={old_metrics.get('kl_from_previous_checkpoint', float('nan')):.4e}" if "kl_from_previous_checkpoint" in old_metrics else "")
                      )
            else:
                result["old_data_size"] = 0

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
