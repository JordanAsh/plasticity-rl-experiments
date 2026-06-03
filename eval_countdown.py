"""
Evaluate a model on Countdown test set using vLLM.
Uses verl's countdown reward function. Greedy decoding.
"""
import argparse
import json
import os
import sys

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "verl-0.4"))
from verl.utils.reward_score.countdown import compute_score as countdown_compute_score


def build_prompts(test_df, tokenizer):
    prompts = []
    for _, row in test_df.iterrows():
        messages = list(row["prompt"])
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)
    return prompts


def evaluate(llm, sampling_params, test_df, tokenizer):
    prompts = build_prompts(test_df, tokenizer)
    ground_truths = [row["reward_model"]["ground_truth"] for _, row in test_df.iterrows()]

    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    format_only = 0
    none = 0
    total = len(outputs)
    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        gt = {"target": int(ground_truths[i]["target"]),
              "numbers": [int(n) for n in ground_truths[i]["numbers"]]}
        score = float(countdown_compute_score(response, gt))
        if score >= 1.0:
            correct += 1
        elif score >= 0.1:
            format_only += 1
        else:
            none += 1
        results.append({
            "response": response[:500],
            "target": gt["target"],
            "numbers": gt["numbers"],
            "score": score,
        })

    return {
        "accuracy": correct / total * 100,
        "format_only": format_only / total * 100,
        "no_answer": none / total * 100,
        "total": total,
    }, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str,
                        default=os.path.expanduser("~/data/countdown/test.parquet"))
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

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

    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=args.max_new_tokens,
                                     stop_token_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>")])

    test_df = pd.read_parquet(args.test_file)
    print(f"\nEvaluating Countdown ({len(test_df)} prompts, greedy)...")
    metrics, results = evaluate(llm, sampling_params, test_df, tokenizer)

    print(f"\n{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"  Correct (score=1.0):    {metrics['accuracy']:.1f}%")
    print(f"  Format only (score=0.1): {metrics['format_only']:.1f}%")
    print(f"  No answer (score=0):    {metrics['no_answer']:.1f}%")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump({"model": args.model_path, **metrics}, f, indent=2)
        with open(os.path.join(args.output_dir, "details.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
