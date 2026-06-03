"""
Evaluate pass@k on Countdown test set using vLLM.
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
from verl.utils.reward_score.countdown import compute_score as countdown_compute_score


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def build_prompts(test_df, tokenizer):
    return [tokenizer.apply_chat_template(list(row["prompt"]), tokenize=False, add_generation_prompt=True)
            for _, row in test_df.iterrows()]


def evaluate_pass_at_k(llm, sampling_params, test_df, tokenizer, k=16):
    prompts = build_prompts(test_df, tokenizer)
    ground_truths = [row["reward_model"]["ground_truth"] for _, row in test_df.iterrows()]
    n = sampling_params.n
    outputs = llm.generate(prompts, sampling_params)

    scores = []
    for i, output in enumerate(outputs):
        gt = {"target": int(ground_truths[i]["target"]),
              "numbers": [int(x) for x in ground_truths[i]["numbers"]]}
        correct = sum(1 for sample in output.outputs
                      if float(countdown_compute_score(sample.text, gt)) >= 1.0)
        scores.append(pass_at_k(n, correct, k))
    return np.mean(scores) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str,
                        default=os.path.expanduser("~/data/countdown/test.parquet"))
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    assert args.n >= args.k
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True, dtype="bfloat16")
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                     max_tokens=args.max_new_tokens, n=args.n,
                                     stop_token_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>")])

    test_df = pd.read_parquet(args.test_file)
    print(f"Evaluating Countdown pass@{args.k} (n={args.n}, temp={args.temperature})...")
    pak = evaluate_pass_at_k(llm, sampling_params, test_df, tokenizer, k=args.k)
    print(f"\nModel: {args.model_path}")
    print(f"Countdown pass@{args.k}: {pak:.1f}%")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump({"model": args.model_path, "n": args.n, "k": args.k,
                       "temperature": args.temperature, f"pass@{args.k}": pak,
                       "total": len(test_df)}, f, indent=2)


if __name__ == "__main__":
    main()
