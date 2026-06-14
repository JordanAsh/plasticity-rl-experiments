#!/usr/bin/env python3
"""
SFT training on positive generations from GRPO training logs, with
plasticity / forgetting instrumentation.

Per-optimizer-step metrics (logged to {output_dir}/metrics/train_metrics.jsonl):
  - global_step, epoch, batch_id (last micro-batch idx), num_supervised_tokens,
    learning_rate
  - batch_loss              (supervised-token-mean loss, accumulated over micro-batches)
  - token_entropy           (mean next-token entropy over supervised tokens)
  - gradient_norm_pre_clip  (returned by clip_grad_norm_)
  - gradient_norm_post_clip
  - parameter_l2_norm                    ||theta_t||
  - parameter_l2_from_init               ||theta_t - theta_0||
  - relative_parameter_l2_from_init      ||theta_t - theta_0|| / ||theta_0||
  - update_norm                          ||theta_t - theta_{t-1}||
  - relative_update_norm                 ||theta_t - theta_{t-1}|| / ||theta_t||

Checkpoints are saved at ~every 10% of total optimizer steps under
{output_dir}/checkpoints/step_{N}/, including model, optimizer, scheduler,
RNG, sampler state, training config, and a per-rank data manifest in
{output_dir}/manifests/rank_{r}.jsonl.

Usage (single GPU):
    python run_sft.py \
        --generation_logs_dir generation_logs/qwen2.5_1.5b_grpo_seed42_20260519_2017 \
        --output_dir sft_outputs/seed42_ordered \
        --ordered

Usage (multi-GPU with DDP):
    torchrun --nproc_per_node=4 run_sft.py \
        --generation_logs_dir generation_logs/qwen2.5_1.5b_grpo_seed42_20260519_2017 \
        --output_dir sft_outputs/seed42_shuffled
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
)
from tqdm import tqdm


def is_dist():
    return dist.is_initialized()

def local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def global_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def is_main():
    return global_rank() == 0

def log(msg):
    if is_main():
        print(msg, flush=True)


# ---------- markers for verl's logged inputs ----------
# verl logs inputs decoded with skip_special_tokens=True from a Qwen-style
# chat template, so the rendered text always contains "\nuser\n{Q}\nassistant\n"
# regardless of what the system prompt happens to be.
USER_MARK = "\nuser\n"
ASSIST_MARK = "\nassistant\n"


def parse_args():
    p = argparse.ArgumentParser(description="SFT on GRPO positive generations")
    # data
    p.add_argument("--generation_logs_dir", type=str, required=True,
                    help="Path to folder with {step}.jsonl files")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B",
                    help="Base model to fine-tune")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Where to save checkpoints and metrics")
    # training
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=4,
                    help="Per-GPU micro-batch size")
    p.add_argument("--effective_batch_size", type=int, default=128,
                    help="Total effective batch size across all GPUs; grad accum is computed automatically")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--schedule", type=str, default="cosine",
                    choices=["cosine", "linear", "constant"])
    p.add_argument("--max_seq_length", type=int, default=3072)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # ordering
    p.add_argument("--ordered", action="store_true",
                    help="Train on positives in step order (default: shuffled)")
    p.add_argument("--score_threshold", type=float, default=0.0,
                    help="Only keep generations with score strictly greater than this (default 0.0)")
    p.add_argument("--drop_all_pass_groups", action="store_true",
                    help="Drop all positives from (step, prompt) groups where every rollout in the "
                         "group passed --score_threshold. Used to break batch homogeneity caused by "
                         "many redundant near-identical templated rollouts of the same prompt.")
    p.add_argument("--max_rl_step", type=int, default=None,
                    help="If set, only load positives from {step}.jsonl files with step <= this value.")
    p.add_argument("--num_epochs", type=int, default=1)
    # instrumentation
    p.add_argument("--num_checkpoints", type=int, default=10,
                    help="Number of checkpoints to save (evenly spaced across training steps).")
    p.add_argument("--save_optimizer_state", action=argparse.BooleanOptionalAction, default=True,
                    help="Save optimizer.pt and scheduler.pt at each checkpoint.")
    p.add_argument("--snapshot_dtype", type=str, default="float32",
                    choices=["float32", "bfloat16", "float16"],
                    help="Dtype for the init/prev parameter snapshots used for norm metrics.")
    p.add_argument("--snapshot_device", type=str, default="cuda",
                    choices=["cuda", "cpu"],
                    help="Where to keep the rank-0 init/prev snapshots. 'cpu' uses pinned host memory, "
                         "freeing ~2x model_size on GPU rank 0 (recommended for 3B+).")
    p.add_argument("--metrics_log_every", type=int, default=1,
                    help="Compute parameter-norm metrics every N optimizer steps (default 1). "
                         "Only the norm metrics are gated; loss/entropy/grad_norm are still per-step.")
    # resume
    p.add_argument("--resume_from", type=str, default=None,
                    help="Path to a checkpoint dir (e.g. <output_dir>/checkpoints/step_100) "
                         "or the literal string 'latest' to auto-pick the highest step in <output_dir>/checkpoints.")
    return p.parse_args()


def load_positives(logs_dir: str, score_threshold: float = 0.0,
                   drop_all_pass_groups: bool = False,
                   max_rl_step: int | None = None) -> list[dict]:
    """Load (prompt, response) positives from per-RL-step .jsonl files.

    Args:
        logs_dir: directory of {step}.jsonl files. Each line has
            {"input": str, "output": str, "score": float, "step": int, "uid": ...}.
        score_threshold: keep only records with score > threshold.
        drop_all_pass_groups: if True, drop all positives from (step, prompt) groups in
            which every rollout passes the score threshold. Such groups are typically
            collapsed-policy rollouts that are nearly identical to each other and cause
            severe batch homogeneity (many duplicated templates per minibatch).
    """
    positives = []
    files = sorted(
        [f for f in os.listdir(logs_dir) if f.endswith(".jsonl")],
        key=lambda x: int(x.split(".")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {logs_dir}")
    if max_rl_step is not None:
        files = [f for f in files if int(f.split(".")[0]) <= max_rl_step]
        if not files:
            raise FileNotFoundError(
                f"No .jsonl files in {logs_dir} have step <= max_rl_step={max_rl_step}"
            )

    skipped = 0
    # If filtering, do a first pass over each file to compute (step, prompt) group sizes
    # and pass-counts. We deliberately key on raw `input` to avoid any tokenization
    # subtleties; it groups exactly the rollouts produced for the same prompt at the
    # same RL step.
    dropped_groups_total = 0
    dropped_positives_total = 0
    for fname in files:
        step = int(fname.split(".")[0])
        path = os.path.join(logs_dir, fname)

        all_pass_inputs: set | None = None
        if drop_all_pass_groups:
            group_total: dict[str, int] = {}
            group_pass: dict[str, int] = {}
            with open(path) as fh:
                for line in fh:
                    record = json.loads(line)
                    inp = record["input"]
                    group_total[inp] = group_total.get(inp, 0) + 1
                    if record["score"] > score_threshold:
                        group_pass[inp] = group_pass.get(inp, 0) + 1
            all_pass_inputs = {
                inp for inp, n in group_total.items()
                if group_pass.get(inp, 0) == n
            }
            dropped_groups_total += len(all_pass_inputs)
            dropped_positives_total += sum(group_total[inp] for inp in all_pass_inputs)

        with open(path) as fh:
            for line_idx, line in enumerate(fh):
                record = json.loads(line)
                if record["score"] <= score_threshold:
                    continue
                inp = record["input"]
                if all_pass_inputs is not None and inp in all_pass_inputs:
                    continue
                # Slice out the last user turn regardless of what system prompt was used
                if not inp.endswith(ASSIST_MARK):
                    skipped += 1
                    continue
                user_start = inp.rfind(USER_MARK, 0, -len(ASSIST_MARK))
                if user_start < 0:
                    skipped += 1
                    continue
                question = inp[user_start + len(USER_MARK):-len(ASSIST_MARK)]
                positives.append({
                    "question": question,
                    "response": record["output"],
                    "step": step,
                    "line_idx": line_idx,
                })

    if skipped > 0:
        print(f"WARNING: Skipped {skipped} records with unexpected template format")
    if drop_all_pass_groups:
        print(f"  drop_all_pass_groups: removed {dropped_groups_total} (step, prompt) groups "
              f"({dropped_positives_total} positives) where every rollout passed the threshold")
    return positives


class SFTDataset(Dataset):
    """SFT dataset. Each item carries a stable sample_id for provenance."""

    def __init__(self, positives: list[dict], tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.items = []
        self.skipped_long = 0

        self.eos_ids = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

        print(f"  Tokenizing {len(positives):,} positives ...")

        prompt_texts = []
        for item in tqdm(positives, desc="  Building prompts"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": item["question"]},
            ]
            prompt_texts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        print("  Batch tokenizing prompts ...")
        prompt_enc = tokenizer(
            prompt_texts, add_special_tokens=False, return_attention_mask=False,
        )

        response_texts = [item["response"] for item in positives]
        print("  Batch tokenizing responses ...")
        response_enc = tokenizer(
            response_texts, add_special_tokens=False, return_attention_mask=False,
        )

        for i in tqdm(range(len(positives)), desc="  Filtering"):
            p_ids = prompt_enc["input_ids"][i]
            r_ids = response_enc["input_ids"][i]
            full_len = len(p_ids) + len(r_ids) + len(self.eos_ids)
            if full_len > max_seq_length:
                self.skipped_long += 1
                continue
            self.items.append({
                "prompt_ids": p_ids,
                "response_ids": r_ids,
                "sample_id": f"{positives[i]['step']}:{positives[i]['line_idx']}",
            })

        if self.skipped_long > 0:
            print(f"  Skipped {self.skipped_long} examples exceeding max_seq_length={max_seq_length}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        prompt_ids = item["prompt_ids"]
        response_ids = item["response_ids"]
        full_ids = torch.tensor(prompt_ids + response_ids + self.eos_ids, dtype=torch.long)
        labels = full_ids.clone()
        labels[:len(prompt_ids)] = -100
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(full_ids),
            "dataset_idx": idx,
            "sample_id": item["sample_id"],
        }


def collate_fn(batch, pad_token_id):
    """Left-pad sequences. Returns tensor batch + provenance lists."""
    max_len = max(ex["input_ids"].size(0) for ex in batch)
    input_ids, labels, attention_mask = [], [], []
    dataset_idx, sample_ids = [], []
    for ex in batch:
        pad_len = max_len - ex["input_ids"].size(0)
        input_ids.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), ex["input_ids"]])
        )
        labels.append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long), ex["labels"]])
        )
        attention_mask.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), ex["attention_mask"]])
        )
        dataset_idx.append(ex["dataset_idx"])
        sample_ids.append(ex["sample_id"])
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
        "dataset_idx": dataset_idx,
        "sample_ids": sample_ids,
    }


# ----------------------------- metric helpers -----------------------------

@torch.no_grad()
def supervised_token_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
) -> tuple[float, int]:
    """Mean next-token entropy over supervised positions, memory-efficient.

    logits:  (B, T, V) — model output (predicts position t+1 from position t).
    labels:  (B, T)   — -100 at non-supervised positions; the next-token target
                       at position t is labels[:, t+1].

    Avoids materializing fp32 (B,T-1,V) tensors: gathers only the supervised
    rows, then computes entropy in fp32 chunks of `chunk_size` rows at a time.
    H(p) = logsumexp(z) - sum_i softmax(z)_i * z_i.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    n = int(mask.sum().item())
    if n == 0:
        return 0.0, 0
    # (n, V) in the model's (bf16) dtype — only supervised rows.
    sel = shift_logits[mask]
    ent_sum = 0.0
    for i in range(0, sel.size(0), chunk_size):
        x = sel[i:i + chunk_size].float()
        lse = torch.logsumexp(x, dim=-1)
        p = torch.softmax(x, dim=-1)
        ent = lse - (p * x).sum(dim=-1)
        ent_sum += float(ent.sum().item())
        del x, lse, p, ent
    return ent_sum, n


def all_reduce_scalars(values: dict) -> dict:
    """Sum-reduce all scalar values across DDP ranks."""
    if not is_dist():
        return values
    keys = sorted(values.keys())
    t = torch.tensor([values[k] for k in keys], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {k: t[i].item() for i, k in enumerate(keys)}


def snapshot_params(model: torch.nn.Module, dtype: torch.dtype,
                    device: torch.device | str | None = None,
                    pin_memory: bool = False) -> list[torch.Tensor]:
    """Clone model params (in order of parameters()) into a list. By default lives on the same device.
    If `device='cpu'` and `pin_memory=True`, snapshots are placed in pinned host memory for fast H2D."""
    out = []
    for p in model.parameters():
        t = p.detach().to(dtype=dtype).clone()
        if device is not None and torch.device(device) != t.device:
            t = t.to(device=device)
        if pin_memory and t.device.type == "cpu":
            t = t.pin_memory()
        out.append(t)
    return out


@torch.no_grad()
def param_l2_norm(params: list[torch.Tensor]) -> float:
    sq = torch.zeros((), dtype=torch.float64, device=params[0].device)
    for p in params:
        sq += p.detach().float().pow(2).sum().to(torch.float64)
    return float(sq.sqrt().item())


@torch.no_grad()
def param_diff_l2_norm(params_a, params_b) -> float:
    """||a - b||_2. Works whether a and b are on the same device or not (streams per-tensor)."""
    a0_dev = params_a[0].device
    b0_dev = params_b[0].device
    sq = torch.zeros((), dtype=torch.float64, device=a0_dev)
    if a0_dev == b0_dev:
        for a, b in zip(params_a, params_b):
            sq += (a.detach().float() - b.detach().float()).pow(2).sum().to(torch.float64)
    else:
        # Stream b -> a's device one tensor at a time.
        for a, b in zip(params_a, params_b):
            b_on_a = b.detach().to(device=a0_dev, non_blocking=True)
            sq += (a.detach().float() - b_on_a.float()).pow(2).sum().to(torch.float64)
            del b_on_a
    return float(sq.sqrt().item())


@torch.no_grad()
def copy_into_snapshot(snapshot: list[torch.Tensor], model: torch.nn.Module):
    """In-place copy current model params into the pre-allocated snapshot list.
    Handles both same-device and cross-device (D2H/H2D) copies."""
    for s, p in zip(snapshot, model.parameters()):
        s.copy_(p.detach().to(dtype=s.dtype), non_blocking=True)


# ----------------------------- checkpointing -----------------------------

def compute_checkpoint_steps(total_steps: int, num_checkpoints: int) -> list[int]:
    if num_checkpoints <= 0:
        return []
    steps = [max(1, round(total_steps * (i + 1) / num_checkpoints)) for i in range(num_checkpoints)]
    return sorted(set(steps))


def save_checkpoint(
    ckpt_dir: str,
    base_model,
    tokenizer,
    optimizer,
    scheduler,
    args,
    global_step: int,
    epoch: int,
    batch_idx: int,
    save_optimizer: bool,
    prev_params: list[torch.Tensor] | None = None,
):
    """Save a checkpoint. Called only on rank 0."""
    os.makedirs(ckpt_dir, exist_ok=True)
    model_dir = os.path.join(ckpt_dir, "model")
    base_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    if save_optimizer:
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))

    rng = {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    torch.save(rng, os.path.join(ckpt_dir, "rng_state.pt"))

    if prev_params is not None:
        torch.save([t.detach().cpu() for t in prev_params],
                   os.path.join(ckpt_dir, "prev_params.pt"))

    train_state = {
        "global_step": global_step,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "config": vars(args),
    }
    with open(os.path.join(ckpt_dir, "train_state.json"), "w") as f:
        json.dump(train_state, f, indent=2)


def resolve_resume_dir(output_dir: str, resume_from: str) -> str:
    """Resolve --resume_from (path or 'latest') to an absolute checkpoint dir."""
    if resume_from == "latest":
        ckpts_root = os.path.join(output_dir, "checkpoints")
        if not os.path.isdir(ckpts_root):
            raise FileNotFoundError(f"No checkpoints dir at {ckpts_root}")
        steps = []
        for d in os.listdir(ckpts_root):
            if d.startswith("step_") and os.path.isdir(os.path.join(ckpts_root, d)):
                try:
                    steps.append(int(d.split("_", 1)[1]))
                except ValueError:
                    continue
        if not steps:
            raise FileNotFoundError(f"No step_* checkpoints in {ckpts_root}")
        return os.path.join(ckpts_root, f"step_{max(steps)}")
    if not os.path.isdir(resume_from):
        raise FileNotFoundError(f"Resume path not found: {resume_from}")
    return resume_from


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---------- DDP setup ----------
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank())

    device = torch.device("cuda", local_rank())

    # ---------- resolve resume checkpoint (if any) ----------
    resume_ckpt_dir = None
    resume_global_step = 0
    resume_epoch = 0
    resume_batch_idx = -1  # next batch to consume = resume_batch_idx + 1
    if args.resume_from:
        resume_ckpt_dir = resolve_resume_dir(args.output_dir, args.resume_from)
        with open(os.path.join(resume_ckpt_dir, "train_state.json")) as fh:
            ts = json.load(fh)
        resume_global_step = int(ts["global_step"])
        resume_epoch = int(ts["epoch"])
        resume_batch_idx = int(ts.get("batch_idx", -1))
        log(f"Resuming from {resume_ckpt_dir}")
        log(f"  global_step={resume_global_step}, epoch={resume_epoch}, batch_idx={resume_batch_idx}")

    # ---------- compute grad accum from effective batch size ----------
    world = world_size()
    if args.effective_batch_size % (world * args.batch_size) != 0:
        raise ValueError(
            f"effective_batch_size ({args.effective_batch_size}) must be divisible by "
            f"world_size * batch_size ({world} * {args.batch_size} = {world * args.batch_size})"
        )
    args.grad_accum_steps = args.effective_batch_size // (world * args.batch_size)
    log(f"  Effective batch size: {args.effective_batch_size} "
        f"({world} GPUs × {args.batch_size} micro-batch × {args.grad_accum_steps} accum)")

    # ---------- output dirs ----------
    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "manifests"), exist_ok=True)

    if is_dist():
        dist.barrier()

    # ---------- load data ----------
    log(f"Loading positives from {args.generation_logs_dir} ...")
    positives = load_positives(
        args.generation_logs_dir,
        score_threshold=args.score_threshold,
        drop_all_pass_groups=args.drop_all_pass_groups,
        max_rl_step=args.max_rl_step,
    )
    log(f"  Total positives: {len(positives):,}")

    if not args.ordered:
        rng = random.Random(args.seed)
        rng.shuffle(positives)
        log("  Mode: shuffled")
    else:
        log("  Mode: ordered by step")

    # ---------- tokenizer & dataset ----------
    log(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = SFTDataset(positives, tokenizer, args.max_seq_length)
    log(f"  Dataset size after filtering: {len(dataset):,}")
    if len(dataset) == 0:
        raise ValueError("No usable SFT examples after filtering")

    # Save dataset index -> sample_id mapping (rank 0 only).
    if is_main():
        with open(os.path.join(args.output_dir, "dataset_samples.jsonl"), "w") as f:
            for i, item in enumerate(dataset.items):
                f.write(json.dumps({"dataset_idx": i, "sample_id": item["sample_id"]}) + "\n")

    if is_main() and len(dataset) > 0:
        ex = dataset[0]
        print("\n--- Sanity check: first example ---")
        prompt_len = (ex["labels"] == -100).sum().item()
        print(f"  Total tokens: {len(ex['input_ids'])}, Prompt tokens (masked): {prompt_len}")
        print(f"  Prompt: {tokenizer.decode(ex['input_ids'][:prompt_len])[:200]}...")
        response_ids = ex["input_ids"][prompt_len:]
        print(f"  Response: {tokenizer.decode(response_ids)[:200]}...")
        print("---\n")

    # ---------- dataloader ----------
    if is_dist():
        sampler = DistributedSampler(
            dataset, shuffle=not args.ordered, seed=args.seed,
        )
    else:
        sampler = SequentialSampler(dataset) if args.ordered else RandomSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    total_steps = (len(dataloader) + args.grad_accum_steps - 1) // args.grad_accum_steps * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    log(f"  Batches/GPU: {len(dataloader)}, Grad accum: {args.grad_accum_steps}")
    log(f"  Total optimizer steps: {total_steps}, Warmup: {warmup_steps}")

    ckpt_steps = compute_checkpoint_steps(total_steps, args.num_checkpoints)
    log(f"  Checkpoint steps: {ckpt_steps}")

    # ---------- model ----------
    model_load_path = (
        os.path.join(resume_ckpt_dir, "model") if resume_ckpt_dir is not None else args.model_path
    )
    log(f"Loading model from {model_load_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    model.to(device)

    if is_dist():
        model = DDP(model, device_ids=[local_rank()])

    base_model = model.module if is_dist() else model

    # ---------- optimizer & scheduler ----------
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [p for n, p in base_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in base_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scheduler = get_scheduler(
        args.schedule, optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Restore optimizer / scheduler / RNG if resuming.
    if resume_ckpt_dir is not None:
        log("  Restoring optimizer, scheduler, and RNG state ...")
        optimizer.load_state_dict(
            torch.load(os.path.join(resume_ckpt_dir, "optimizer.pt"), map_location=device)
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(resume_ckpt_dir, "scheduler.pt"), map_location="cpu")
        )
        rng = torch.load(os.path.join(resume_ckpt_dir, "rng_state.pt"), map_location="cpu")
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    # ---------- snapshots for norm metrics (rank 0 only — params replicated) ----------
    snapshot_dtype = {
        "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16,
    }[args.snapshot_dtype]
    snap_dev = torch.device(args.snapshot_device)
    snap_pin = (snap_dev.type == "cpu")

    init_params = None
    prev_params = None
    init_norm = None
    init_params_path = os.path.join(args.output_dir, "init_params.pt")
    if is_main():
        if resume_ckpt_dir is not None:
            log(f"  Loading init snapshot from {init_params_path} -> {snap_dev}")
            loaded = torch.load(init_params_path, map_location="cpu")
            init_params = [t.to(device=snap_dev, dtype=snapshot_dtype) for t in loaded]
            if snap_pin:
                init_params = [t.pin_memory() for t in init_params]
            prev_path = os.path.join(resume_ckpt_dir, "prev_params.pt")
            log(f"  Loading prev snapshot from {prev_path} -> {snap_dev}")
            loaded = torch.load(prev_path, map_location="cpu")
            prev_params = [t.to(device=snap_dev, dtype=snapshot_dtype) for t in loaded]
            if snap_pin:
                prev_params = [t.pin_memory() for t in prev_params]
        else:
            log(f"  Building parameter snapshots in {args.snapshot_dtype} on {snap_dev} (init + prev) ...")
            init_params = snapshot_params(base_model, snapshot_dtype, device=snap_dev, pin_memory=snap_pin)
            prev_params = snapshot_params(base_model, snapshot_dtype, device=snap_dev, pin_memory=snap_pin)
            torch.save([t.detach().cpu() for t in init_params], init_params_path)
        init_norm = param_l2_norm(init_params)
        log(f"  ||theta_0|| = {init_norm:.4e}")

    # ---------- metrics file ----------
    metrics_path = os.path.join(args.output_dir, "metrics", "train_metrics.jsonl")
    metrics_fh = open(metrics_path, "a") if is_main() else None

    # Per-rank manifest of every batch trained on.
    manifest_path = os.path.join(args.output_dir, "manifests", f"rank_{global_rank()}.jsonl")
    manifest_fh = open(manifest_path, "a")

    # Save config once.
    if is_main():
        with open(os.path.join(args.output_dir, "sft_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # ---------- training loop ----------
    model.train()
    global_step = resume_global_step
    optimizer.zero_grad()

    start_epoch = resume_epoch
    start_batch_idx_in_epoch = resume_batch_idx + 1  # next unconsumed batch
    log(f"  Starting at global_step={global_step}, epoch={start_epoch}, batch_idx={start_batch_idx_in_epoch}")

    # Per-optimizer-step accumulators.
    accum_loss_tok_sum = 0.0
    accum_ent_tok_sum = 0.0
    accum_tok_count = 0
    accum_batch_ids = []
    accum_dataset_indices = []
    accum_sample_ids = []

    for epoch in range(start_epoch, args.num_epochs):
        if is_dist() and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        skip_n = start_batch_idx_in_epoch if epoch == start_epoch else 0
        if skip_n > 0:
            log(f"  Skipping first {skip_n} batches of epoch {epoch} (resume) ...")
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            initial=skip_n,
            desc=f"Epoch {epoch+1}",
            disable=not is_main(),
        )
        for batch_idx, batch in pbar:
            if batch_idx < skip_n:
                continue
            ds_idx = batch.pop("dataset_idx")
            s_ids = batch.pop("sample_ids")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            outputs = model(**batch)

            ent_sum, n_sup = supervised_token_entropy(outputs.logits, batch["labels"])

            is_accum_boundary = (batch_idx + 1) % args.grad_accum_steps == 0
            is_last_batch = (batch_idx + 1) == len(dataloader)
            if is_last_batch and not is_accum_boundary:
                actual_accum = (batch_idx % args.grad_accum_steps) + 1
            else:
                actual_accum = args.grad_accum_steps

            loss = outputs.loss / actual_accum
            loss.backward()

            # outputs.loss is mean over supervised tokens for this micro-batch.
            accum_loss_tok_sum += outputs.loss.item() * n_sup
            accum_ent_tok_sum += ent_sum
            accum_tok_count += n_sup
            accum_batch_ids.append(batch_idx)
            accum_dataset_indices.append(list(ds_idx))
            accum_sample_ids.append(list(s_ids))

            del outputs, loss

            if is_accum_boundary or is_last_batch:
                # gradient norm pre-clip (returned by clip_grad_norm_)
                grad_norm_pre = torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), args.grad_clip
                )
                grad_norm_pre = float(
                    grad_norm_pre.item() if torch.is_tensor(grad_norm_pre) else grad_norm_pre
                )
                # post-clip global norm (cheap recomputation)
                with torch.no_grad():
                    sq = torch.zeros((), dtype=torch.float64, device=device)
                    for p in base_model.parameters():
                        if p.grad is not None:
                            sq += p.grad.detach().float().pow(2).sum().to(torch.float64)
                    grad_norm_post = float(sq.sqrt().item())

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # aggregate cheap per-token stats across ranks
                agg = all_reduce_scalars({
                    "loss_tok_sum": accum_loss_tok_sum,
                    "ent_tok_sum": accum_ent_tok_sum,
                    "tok_count": float(accum_tok_count),
                })
                tok_count = max(1.0, agg["tok_count"])
                batch_loss = agg["loss_tok_sum"] / tok_count
                token_entropy = agg["ent_tok_sum"] / tok_count

                metric_record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "batch_id": batch_idx,
                    "num_supervised_tokens": int(agg["tok_count"]),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "batch_loss": batch_loss,
                    "token_entropy": token_entropy,
                    "gradient_norm_pre_clip": grad_norm_pre,
                    "gradient_norm_post_clip": grad_norm_post,
                }
                if is_main():
                    # Gate the expensive norm metrics on metrics_log_every. Also always compute
                    # them on the very last step. The cheap loss/entropy/grad fields above are
                    # already populated unconditionally.
                    do_norms = (
                        global_step % args.metrics_log_every == 0
                        or global_step == total_steps
                    )
                    if do_norms:
                        cur_norm = param_l2_norm([p.detach() for p in base_model.parameters()])
                        diff_init = param_diff_l2_norm(
                            [p.detach() for p in base_model.parameters()], init_params
                        )
                        diff_prev = param_diff_l2_norm(
                            [p.detach() for p in base_model.parameters()], prev_params
                        )
                        metric_record.update({
                            "parameter_l2_norm": cur_norm,
                            "parameter_l2_from_init": diff_init,
                            "relative_parameter_l2_from_init": diff_init / max(init_norm, 1e-12),
                            "update_norm": diff_prev,
                            "relative_update_norm": diff_prev / max(cur_norm, 1e-12),
                        })
                        copy_into_snapshot(prev_params, base_model)

                    metrics_fh.write(json.dumps(metric_record) + "\n")
                    metrics_fh.flush()

                    if global_step % 10 == 0 or global_step == total_steps:
                        pbar.set_postfix(
                            step=f"{global_step}/{total_steps}",
                            loss=f"{batch_loss:.4f}",
                            ent=f"{token_entropy:.3f}",
                            gn=f"{grad_norm_pre:.2f}",
                        )
                        print(
                            f"  step {global_step}/{total_steps} | "
                            f"loss: {batch_loss:.4f} | ent: {token_entropy:.3f} | "
                            f"gn_pre: {grad_norm_pre:.3f} | gn_post: {grad_norm_post:.3f} | "
                            f"||θ-θ0||: {metric_record.get('parameter_l2_from_init', float('nan')):.3e} | "
                            f"||Δθ||: {metric_record.get('update_norm', float('nan')):.3e} | "
                            f"lr: {scheduler.get_last_lr()[0]:.2e}"
                        )

                # per-rank manifest entry for this optimizer step
                manifest_entry = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "rank": global_rank(),
                    "world_size": world,
                    "micro_batches": [
                        {"batch_id": bid, "dataset_idx": didx, "sample_ids": sids}
                        for bid, didx, sids in zip(
                            accum_batch_ids, accum_dataset_indices, accum_sample_ids
                        )
                    ],
                }
                manifest_fh.write(json.dumps(manifest_entry) + "\n")
                manifest_fh.flush()

                accum_loss_tok_sum = 0.0
                accum_ent_tok_sum = 0.0
                accum_tok_count = 0
                accum_batch_ids = []
                accum_dataset_indices = []
                accum_sample_ids = []

                if global_step in ckpt_steps:
                    if is_dist():
                        dist.barrier()
                    if is_main():
                        ckpt_dir = os.path.join(
                            args.output_dir, "checkpoints", f"step_{global_step}"
                        )
                        log(f"  Saving checkpoint -> {ckpt_dir}")
                        save_checkpoint(
                            ckpt_dir,
                            base_model,
                            tokenizer,
                            optimizer,
                            scheduler,
                            args,
                            global_step,
                            epoch,
                            batch_idx,
                            args.save_optimizer_state,
                            prev_params=prev_params,
                        )
                    if is_dist():
                        dist.barrier()

    if is_main():
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        base_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"\nFinal model saved to {final_dir}")
        if metrics_fh is not None:
            metrics_fh.close()
        print("Done.")

    manifest_fh.close()

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
