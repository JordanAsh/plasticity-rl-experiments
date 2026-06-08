"""Evaluate pass@k on Knights and Knaves test set using vLLM."""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "verl_extensions"))
from reward_score.kk import compute_score as kk_compute_score


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def build_prompts(test_df, tokenizer):
    return [tokenizer.apply_chat_template(list(row["prompt"]), tokenize=False, add_generation_prompt=True)
            for _, row in test_df.iterrows()]


def evaluate_pass_at_k(llm, sampling_params, test_df, tokenizer, ks):
    """Run vLLM once with n samples per prompt, then compute pass@k for every k in `ks`.

    Returns:
        results: {k: {"mean": ..., "by_num_people": {...}}, ...}
        details: list[dict] one entry per prompt with all n samples + per-sample correctness.
    """
    prompts = build_prompts(test_df, tokenizer)
    ground_truths = [row["reward_model"]["ground_truth"] for _, row in test_df.iterrows()]
    n_people = [row["extra_info"].get("num_people", 0) for _, row in test_df.iterrows()]
    n = sampling_params.n
    outputs = llm.generate(prompts, sampling_params)

    # Per-prompt counts of correct samples (out of n)
    correct_per_prompt = []
    np_per_prompt = []
    details = []
    for i, output in enumerate(outputs):
        gt = ground_truths[i]
        sample_records = []
        c = 0
        for sample in output.outputs:
            score = float(kk_compute_score(sample.text, gt))
            is_correct = score >= 2.5
            if is_correct:
                c += 1
            sample_records.append({
                "text": sample.text,
                "score": score,
                "correct": is_correct,
            })
        correct_per_prompt.append(c)
        np_per_prompt.append(int(n_people[i]) if n_people[i] else 0)
        details.append({
            "prompt": prompts[i],
            "ground_truth": gt,
            "num_people": np_per_prompt[-1],
            "n": n,
            "num_correct": c,
            "samples": sample_records,
        })

    results = {}
    for k in ks:
        scores = [pass_at_k(n, c, k) for c in correct_per_prompt]
        by_n = {}
        for npl, s in zip(np_per_prompt, scores):
            by_n.setdefault(npl, []).append(s)
        results[k] = {
            "mean": float(np.mean(scores)) * 100,
            "by_num_people": {str(np_): {"pak": float(np.mean(v)) * 100, "n": len(v)}
                              for np_, v in sorted(by_n.items())},
        }
        # Stamp per-prompt pass@k onto details so they're directly inspectable.
        for d, s in zip(details, scores):
            d[f"pass@{k}"] = float(s)
    return results, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str,
                        default=os.path.expanduser("~/data/kk/test.parquet"))
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--k", type=int, default=None,
                        help="Single k to evaluate. Ignored if --ks is set.")
    parser.add_argument("--ks", type=str, default=None,
                        help="Comma-separated list of k values to evaluate from the same n samples, e.g. '1,8,16,32'.")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.ks:
        ks = [int(x) for x in args.ks.split(",") if x.strip()]
    elif args.k is not None:
        ks = [args.k]
    else:
        raise SystemExit("Must specify either --k or --ks")
    assert max(ks) <= args.n, f"max(ks)={max(ks)} but n={args.n}"

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
    print(f"Evaluating KK pass@{ks} (n={args.n}, temp={args.temperature})...")
    results, details = evaluate_pass_at_k(llm, sampling_params, test_df, tokenizer, ks=ks)
    print(f"\nModel: {args.model_path}")
    for k in ks:
        print(f"KK pass@{k}: {results[k]['mean']:.1f}%")
        for npl, v in results[k]["by_num_people"].items():
            print(f"  {npl}ppl: {v['pak']:.1f}% ({v['n']} examples)")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        summary = {
            "model": args.model_path,
            "n": args.n,
            "ks": ks,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        for k in ks:
            summary[f"pass@{k}"] = results[k]["mean"]
            summary[f"pass@{k}_by_num_people"] = results[k]["by_num_people"]
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(args.output_dir, "details.json"), "w") as f:
            json.dump(details, f, indent=2)


if __name__ == "__main__":
    main()
