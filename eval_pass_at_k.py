"""
Evaluate pass@k for models on GSM8K and MATH test sets using vLLM.
Uses the same chat template and reward functions as verl.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "verl-0.4"))
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score
from verl.utils.reward_score.math import compute_score as math_compute_score


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Codex paper, Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def build_prompts(test_df, tokenizer):
    prompts = []
    for _, row in test_df.iterrows():
        messages = list(row["prompt"])
        # Qwen2.5 chat template auto-adds system prompt if missing
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)
    return prompts


def evaluate_pass_at_k(llm, sampling_params, test_df, tokenizer, compute_score_fn, k=16):
    prompts = build_prompts(test_df, tokenizer)
    ground_truths = [row["reward_model"]["ground_truth"] for _, row in test_df.iterrows()]
    n = sampling_params.n

    outputs = llm.generate(prompts, sampling_params)

    pass_at_k_scores = []
    greedy_scores = []
    per_problem = []

    for i, output in enumerate(outputs):
        gt = ground_truths[i]
        correct = 0
        for sample in output.outputs:
            score = compute_score_fn(sample.text, gt)
            if isinstance(score, dict):
                score = score.get("score", 0.0)
            correct += int(float(score) > 0.5)

        pak = pass_at_k(n, correct, k)
        pass_at_k_scores.append(pak)
        per_problem.append({"correct": correct, "total": n, f"pass@{k}": pak})

    mean_pass_at_k = np.mean(pass_at_k_scores) * 100
    return mean_pass_at_k, per_problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gsm8k_test", type=str, default=os.path.expanduser("~/data/gsm8k/test.parquet"))
    parser.add_argument("--math_test", type=str, default=os.path.expanduser("~/data/math/test.parquet"))
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n", type=int, default=16, help="Number of samples per prompt (must be >= k)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    assert args.n >= args.k, f"n ({args.n}) must be >= k ({args.k})"

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.n,
    )

    results_summary = {}

    if os.path.exists(args.gsm8k_test):
        print(f"\n{'='*50}")
        print(f"Evaluating GSM8K pass@{args.k} (n={args.n}, temp={args.temperature})...")
        gsm8k_df = pd.read_parquet(args.gsm8k_test)
        gsm8k_pass_at_k, gsm8k_details = evaluate_pass_at_k(
            llm, sampling_params, gsm8k_df, tokenizer, gsm8k_compute_score, k=args.k
        )
        print(f"GSM8K pass@{args.k}: {gsm8k_pass_at_k:.1f}%")
        results_summary["gsm8k"] = {f"pass@{args.k}": gsm8k_pass_at_k, "total": len(gsm8k_df)}

    if os.path.exists(args.math_test):
        print(f"\n{'='*50}")
        print(f"Evaluating MATH pass@{args.k} (n={args.n}, temp={args.temperature})...")
        math_df = pd.read_parquet(args.math_test)
        math_pass_at_k, math_details = evaluate_pass_at_k(
            llm, sampling_params, math_df, tokenizer, math_compute_score, k=args.k
        )
        print(f"MATH pass@{args.k}: {math_pass_at_k:.1f}%")
        results_summary["math"] = {f"pass@{args.k}": math_pass_at_k, "total": len(math_df)}

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"Sampling: n={args.n}, temp={args.temperature}, top_p={args.top_p}")
    for dataset, res in results_summary.items():
        print(f"  {dataset.upper():>8} pass@{args.k}: {res[f'pass@{args.k}']:.1f}%")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump({"model": args.model_path, "n": args.n, "k": args.k,
                       "temperature": args.temperature, **results_summary}, f, indent=2)
        print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
