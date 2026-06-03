"""Preprocess Knights and Knaves dataset into verl parquet format.

Loads from HuggingFace K-and-K/knights-and-knaves, concatenates 3-7ppl train
subsets (Logic-RL's setup) and 2-8ppl test subsets for evaluation.
"""
import argparse
import os

from datasets import load_dataset, concatenate_datasets

INSTRUCTION = (
    "You first think about the reasoning process in your mind and then "
    "provide the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. "
    "List the identity of each person one by one, for example, "
    "<answer> (1) Zoey is a knight\\n(2) Oliver is a knight\\n(3) ... </answer>.\n\n"
)


def make_user_content(quiz: str) -> str:
    return INSTRUCTION + quiz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", default=os.path.expanduser("~/data/kk"))
    p.add_argument("--train_subsets", nargs="+",
                    default=["3ppl", "4ppl", "5ppl", "6ppl", "7ppl"])
    p.add_argument("--test_subsets", nargs="+",
                    default=["2ppl", "3ppl", "4ppl", "5ppl", "6ppl", "7ppl", "8ppl"])
    args = p.parse_args()

    train_parts = [load_dataset("K-and-K/knights-and-knaves", "train", split=s)
                   for s in args.train_subsets]
    test_parts = [load_dataset("K-and-K/knights-and-knaves", "test", split=s)
                  for s in args.test_subsets]

    train_ds = concatenate_datasets(train_parts)
    test_ds = concatenate_datasets(test_parts)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    def map_fn(split):
        def fn(example, idx):
            return {
                "data_source": "kk_logic",
                "prompt": [{"role": "user", "content": make_user_content(example["quiz"])}],
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "solution_text_format": example["solution_text_format"],
                        "statements": example.get("statements", ""),
                    },
                },
                "extra_info": {"split": split, "index": idx,
                                "num_people": len(example.get("names", []))},
            }
        return fn

    train_ds = train_ds.map(map_fn("train"), with_indices=True,
                             remove_columns=train_ds.column_names)
    test_ds = test_ds.map(map_fn("test"), with_indices=True,
                            remove_columns=test_ds.column_names)

    os.makedirs(args.local_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_ds.to_parquet(os.path.join(args.local_dir, "test.parquet"))
    print(f"Saved to {args.local_dir}")


if __name__ == "__main__":
    main()
