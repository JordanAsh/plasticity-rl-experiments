#!/usr/bin/env python3
"""
SFT training on positive generations from GRPO training logs.

Extracts all generations with score > 0 from a generation_logs directory,
reconstructs the exact chat template used during RL training, and trains
Qwen2.5-1.5B for one epoch.

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
import sys
from pathlib import Path

import torch
import torch.distributed as dist
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

def is_main():
    return local_rank() == 0

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
                    help="Where to save the final checkpoint")
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
    # ordering
    p.add_argument("--ordered", action="store_true",
                    help="Train on positives in step order (default: shuffled)")
    p.add_argument("--score_threshold", type=float, default=0.0,
                    help="Only keep generations with score strictly greater than this (default 0.0)")
    p.add_argument("--num_epochs", type=int, default=1)
    return p.parse_args()


def load_positives(logs_dir: str, score_threshold: float = 0.0) -> list[dict]:
    """Load positive generations from JSONL files, preserving step order.

    A generation is "positive" if its score > score_threshold.
    """
    positives = []
    files = sorted(
        [f for f in os.listdir(logs_dir) if f.endswith(".jsonl")],
        key=lambda x: int(x.split(".")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {logs_dir}")

    skipped = 0
    for fname in files:
        step = int(fname.split(".")[0])
        with open(os.path.join(logs_dir, fname)) as fh:
            for line_idx, line in enumerate(fh):
                record = json.loads(line)
                if record["score"] <= score_threshold:
                    continue
                inp = record["input"]
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
    return positives


class SFTDataset(Dataset):
    """Dataset that reconstructs exact chat-templated sequences for SFT.

    Uses batch tokenization for speed. Constructs input_ids as
    prompt_ids + response_ids to guarantee a correct label-masking boundary.
    """

    def __init__(self, positives: list[dict], tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.items = []
        self.skipped_long = 0

        self.eos_ids = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

        # --- Batch tokenize prompts and responses for speed ---
        print(f"  Tokenizing {len(positives):,} positives ...")

        # Build prompt texts via chat template (this is fast, no tokenization)
        prompt_texts = []
        for item in tqdm(positives, desc="  Building prompts"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": item["question"]},
            ]
            prompt_texts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        # Batch tokenize prompts
        print("  Batch tokenizing prompts ...")
        prompt_enc = tokenizer(
            prompt_texts, add_special_tokens=False, return_attention_mask=False,
        )

        # Batch tokenize responses
        response_texts = [item["response"] for item in positives]
        print("  Batch tokenizing responses ...")
        response_enc = tokenizer(
            response_texts, add_special_tokens=False, return_attention_mask=False,
        )

        # Filter by length and store
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
            })

        if self.skipped_long > 0:
            print(f"  Skipped {self.skipped_long} examples exceeding max_seq_length={max_seq_length}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        prompt_ids = item["prompt_ids"]
        response_ids = item["response_ids"]

        # Concatenate: prompt + response + <|im_end|>
        full_ids = torch.tensor(prompt_ids + response_ids + self.eos_ids, dtype=torch.long)
        labels = full_ids.clone()
        labels[:len(prompt_ids)] = -100

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(full_ids),
        }


def collate_fn(batch, pad_token_id):
    """Left-pad sequences to the same length within a batch."""
    max_len = max(ex["input_ids"].size(0) for ex in batch)
    input_ids = []
    labels = []
    attention_mask = []
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
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---------- DDP setup ----------
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank())

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

    # ---------- load data ----------
    log(f"Loading positives from {args.generation_logs_dir} ...")
    positives = load_positives(args.generation_logs_dir, score_threshold=args.score_threshold)
    log(f"  Total positives: {len(positives):,}")

    if not args.ordered:
        import random
        random.seed(args.seed)
        random.shuffle(positives)
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

    # Sanity check: print a reconstructed example
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

    # ---------- model ----------
    log(f"Loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    device = torch.device("cuda", local_rank())
    model.to(device)

    if is_dist():
        model = DDP(model, device_ids=[local_rank()])

    # ---------- optimizer & scheduler ----------
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    base_model = model.module if is_dist() else model
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

    # ---------- training loop ----------
    model.train()
    global_step = 0
    accum_loss = 0.0
    accum_count = 0
    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        if is_dist() and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}",
                    disable=not is_main())
        for batch_idx, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            # Determine actual accumulation count for final partial window
            is_accum_boundary = (batch_idx + 1) % args.grad_accum_steps == 0
            is_last_batch = (batch_idx + 1) == len(dataloader)
            if is_last_batch and not is_accum_boundary:
                actual_accum = (batch_idx % args.grad_accum_steps) + 1
            else:
                actual_accum = args.grad_accum_steps

            loss = outputs.loss / actual_accum
            loss.backward()
            accum_loss += outputs.loss.item()
            accum_count += 1

            if is_accum_boundary or is_last_batch:
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main() and (global_step % 10 == 0 or global_step == total_steps):
                    avg_loss = accum_loss / accum_count
                    pbar.set_postfix(
                        step=f"{global_step}/{total_steps}",
                        loss=f"{avg_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    print(
                        f"  step {global_step}/{total_steps} | "
                        f"loss: {avg_loss:.4f} | "
                        f"lr: {scheduler.get_last_lr()[0]:.2e}"
                    )
                accum_loss = 0.0
                accum_count = 0

    # ---------- save (rank 0 only) ----------
    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)
        base_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\nModel saved to {args.output_dir}")

        with open(os.path.join(args.output_dir, "sft_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print("Done.")

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
