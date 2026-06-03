"""Evaluate a model on Knights and Knaves test set using vLLM.

Uses verl's kk_logic reward function. Greedy decoding.
"""
import argparse
import json
import os
import sys

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "verl-0.4"))
from verl.utils.reward_score.kk import compute_score as kk_compute_score


def build_prompts(test_df, tokenizer):
    return [tokenizer.apply_chat_template(list(row["prompt"]), tokenize=False, add_generation_prompt=True)
            for _, row in test_df.iterrows()]


def evaluate(llm, sampling_params, test_df, tokenizer):
    prompts = build_prompts(test_df, tokenizer)
    ground_truths = [row["reward_model"]["ground_truth"] for _, row in test_df.iterrows()]
    n_people = [row["extra_info"].get("num_people", 0) for _, row in test_df.iterrows()]
    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    format_only = 0
    by_n = {}
    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        score = float(kk_compute_score(response, ground_truths[i]))
        # Logic-RL scoring: 3 = full correct, +0.5 = format-only with wrong answer parseable, -0.5 broken format correct, -3 = no answer
        is_correct = score >= 2.5  # only +3 counts as fully correct
        if is_correct:
            correct += 1
        elif score > 0:
            format_only += 1
        np_ = int(n_people[i]) if n_people[i] else 0
        by_n.setdefault(np_, {"correct": 0, "total": 0})
        by_n[np_]["total"] += 1
        if is_correct:
            by_n[np_]["correct"] += 1
        results.append({"score": score, "response": response[:500],
                        "n_people": np_, "is_correct": is_correct})

    total = len(outputs)
    accuracy = correct / total * 100
    metrics = {
        "accuracy": accuracy,
        "format_only_pct": format_only / total * 100,
        "total": total,
        "by_num_people": {str(n): {"acc": v["correct"]/v["total"]*100, "n": v["total"]}
                          for n, v in sorted(by_n.items())},
    }
    return metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str,
                        default=os.path.expanduser("~/data/kk/test.parquet"))
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True, dtype="bfloat16")
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=args.max_new_tokens,
                                     stop_token_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>")])

    test_df = pd.read_parquet(args.test_file)
    print(f"\nEvaluating KK ({len(test_df)} prompts, greedy)...")
    metrics, results = evaluate(llm, sampling_params, test_df, tokenizer)

    print(f"\n{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"  Overall accuracy:   {metrics['accuracy']:.1f}%")
    print(f"  Format-only (no correct answer): {metrics['format_only_pct']:.1f}%")
    print(f"  By # people:")
    for n, v in metrics["by_num_people"].items():
        print(f"    {n}ppl: {v['acc']:.1f}% ({v['n']} examples)")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump({"model": args.model_path, **metrics}, f, indent=2)
        with open(os.path.join(args.output_dir, "details.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
