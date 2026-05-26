"""
Evaluate a model on GSM8K and MATH test sets using vLLM.
Uses the same chat template, system prompt, and reward functions as verl.
"""
import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Add verl to path for reward functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "verl-0.4"))
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score
from verl.utils.reward_score.math import compute_score as math_compute_score

SYSTEM_PROMPT = "You are a helpful assistant."


def build_prompts(test_df, tokenizer):
    """Build prompts using the same chat template as verl."""
    prompts = []
    for _, row in test_df.iterrows():
        # row["prompt"] is a list of message dicts, e.g. [{"role": "user", "content": "..."}]
        messages = list(row["prompt"])
        # verl's rl_dataset.py prepends system message if not present
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)
    return prompts


def evaluate_dataset(llm, sampling_params, test_df, tokenizer, data_source, compute_score_fn):
    """Generate and score predictions for a dataset."""
    prompts = build_prompts(test_df, tokenizer)
    ground_truths = [row["reward_model"]["ground_truth"] for _, row in test_df.iterrows()]

    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    total = len(outputs)
    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        gt = ground_truths[i]
        score = compute_score_fn(response, gt)
        if isinstance(score, dict):
            score = score.get("score", 0.0)
        score = float(score)
        correct += score
        results.append({
            "prompt": prompts[i][:200],
            "response": response[:500],
            "ground_truth": gt,
            "score": score,
        })

    accuracy = correct / total * 100
    return accuracy, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gsm8k_test", type=str, default=os.path.expanduser("~/data/gsm8k/test.parquet"))
    parser.add_argument("--math_test", type=str, default=os.path.expanduser("~/data/math/test.parquet"))
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save detailed results (optional)")
    args = parser.parse_args()

    # Must set before importing vllm
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

    # Greedy decoding — matches verl's val_kwargs
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    results_summary = {}

    # GSM8K
    if os.path.exists(args.gsm8k_test):
        print(f"\n{'='*50}")
        print("Evaluating GSM8K...")
        gsm8k_df = pd.read_parquet(args.gsm8k_test)
        gsm8k_acc, gsm8k_results = evaluate_dataset(
            llm, sampling_params, gsm8k_df, tokenizer,
            "openai/gsm8k", gsm8k_compute_score
        )
        print(f"GSM8K accuracy: {gsm8k_acc:.1f}% ({int(gsm8k_acc * len(gsm8k_df) / 100)}/{len(gsm8k_df)})")
        results_summary["gsm8k"] = {"accuracy": gsm8k_acc, "total": len(gsm8k_df)}
    else:
        print(f"GSM8K test file not found: {args.gsm8k_test}")

    # MATH
    if os.path.exists(args.math_test):
        print(f"\n{'='*50}")
        print("Evaluating MATH...")
        math_df = pd.read_parquet(args.math_test)
        math_acc, math_results = evaluate_dataset(
            llm, sampling_params, math_df, tokenizer,
            "DigitalLearningGmbH/MATH-lighteval", math_compute_score
        )
        print(f"MATH accuracy: {math_acc:.1f}% ({int(math_acc * len(math_df) / 100)}/{len(math_df)})")
        results_summary["math"] = {"accuracy": math_acc, "total": len(math_df)}
    else:
        print(f"MATH test file not found: {args.math_test}")

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Model: {args.model_path}")
    for dataset, res in results_summary.items():
        print(f"  {dataset.upper():>8}: {res['accuracy']:.1f}%")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump({"model": args.model_path, **results_summary}, f, indent=2)
        if "gsm8k" in results_summary:
            with open(os.path.join(args.output_dir, "gsm8k_details.json"), "w") as f:
                json.dump(gsm8k_results, f, indent=2)
        if "math" in results_summary:
            with open(os.path.join(args.output_dir, "math_details.json"), "w") as f:
                json.dump(math_results, f, indent=2)
        print(f"\nDetailed results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
