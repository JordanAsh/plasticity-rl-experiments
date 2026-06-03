"""Preprocess Countdown-Tasks-3to4 into verl parquet format.

Adapted from https://github.com/Jiayi-Pan/TinyZero
"""
import argparse
import os

from datasets import load_dataset


def make_prefix(dp):
    target = dp["target"]
    numbers = dp["nums"]
    return (
        f"Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only "
        "be used once. Show your work in <think> </think> tags. And return the final "
        "answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", default=os.path.expanduser("~/data/countdown"))
    p.add_argument("--train_size", type=int, default=327680)
    p.add_argument("--test_size", type=int, default=1024)
    args = p.parse_args()

    raw = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    assert len(raw) > args.train_size + args.test_size, \
        f"raw has {len(raw)}, need {args.train_size + args.test_size}"

    train = raw.select(range(args.train_size))
    test = raw.select(range(args.train_size, args.train_size + args.test_size))

    def make_map_fn(split):
        def process_fn(example, idx):
            return {
                "data_source": "countdown",
                "prompt": [{"role": "user", "content": make_prefix(example)}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": example["target"], "numbers": example["nums"]},
                },
                "extra_info": {"split": split, "index": idx},
            }
        return process_fn

    train = train.map(function=make_map_fn("train"), with_indices=True)
    test = test.map(function=make_map_fn("test"), with_indices=True)

    os.makedirs(args.local_dir, exist_ok=True)
    train.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test.to_parquet(os.path.join(args.local_dir, "test.parquet"))
    print(f"Saved {len(train)} train + {len(test)} test to {args.local_dir}")


if __name__ == "__main__":
    main()
