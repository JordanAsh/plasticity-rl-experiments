#!/usr/bin/env python3
"""
Plot training-time and checkpoint-time metrics for an SFT run.

Reads:
  {run_dir}/metrics/train_metrics.jsonl
  {run_dir}/checkpoints/step_*/eval_metrics.json   (optional)

Writes:
  {run_dir}/plots/train/<metric>.png
  {run_dir}/plots/checkpoints/<metric>.png
  {run_dir}/plots/checkpoints/dead_units_per_layer_step_<N>.png

Usage:
    python plot_metrics.py --run_dir sft_outputs/seed42_shuffled
"""

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_METRICS = [
    "batch_loss",
    "token_entropy",
    "gradient_norm_pre_clip",
    "gradient_norm_post_clip",
    "parameter_l2_norm",
    "parameter_l2_from_init",
    "relative_parameter_l2_from_init",
    "update_norm",
    "relative_update_norm",
    "learning_rate",
    "num_supervised_tokens",
]

CKPT_SCALAR_FIELDS = [
    "loss",
    "token_entropy",
    "kl_from_init",
    "kl_from_previous_checkpoint",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--log_y", action="store_true",
                   help="Use log scale for grad norms / KL plots.")
    return p.parse_args()


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def plot_train(run_dir: str, log_y: bool):
    metrics_path = os.path.join(run_dir, "metrics", "train_metrics.jsonl")
    if not os.path.exists(metrics_path):
        print(f"No {metrics_path}; skipping training plots.")
        return
    rows = read_jsonl(metrics_path)
    if not rows:
        return
    out_dir = os.path.join(run_dir, "plots", "train")
    os.makedirs(out_dir, exist_ok=True)

    steps = [r["global_step"] for r in rows]
    for key in TRAIN_METRICS:
        ys = [r.get(key) for r in rows]
        if all(v is None for v in ys):
            continue
        ys_clean = [(s, y) for s, y in zip(steps, ys) if y is not None]
        xs, vals = zip(*ys_clean)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, vals, lw=1.0)
        ax.set_xlabel("global_step")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
        if log_y and ("grad" in key or "kl" in key or "update_norm" == key):
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{key}.png"), dpi=120)
        plt.close(fig)
    print(f"Wrote training plots to {out_dir}")


def plot_checkpoints(run_dir: str, log_y: bool):
    ckpt_files = sorted(
        glob(os.path.join(run_dir, "checkpoints", "step_*", "eval_metrics.json")),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]),
    )
    if not ckpt_files:
        print("No eval_metrics.json files found; skipping checkpoint plots.")
        return

    rows = [json.load(open(p)) for p in ckpt_files]
    out_dir = os.path.join(run_dir, "plots", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    steps = [r["global_step"] for r in rows]

    # ---------- scalar fields, probe vs old_data ----------
    for key in CKPT_SCALAR_FIELDS:
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for split in ("probe", "old_data"):
            xs, ys = [], []
            for s, r in zip(steps, rows):
                v = r.get(split, {}).get(key)
                if v is not None:
                    xs.append(s)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, marker="o", label=split)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("global_step")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if log_y and "kl" in key:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{key}.png"), dpi=120)
        plt.close(fig)

    # ---------- gradient diagnostics ----------
    for grad_key in ("mean_gradient_norm", "gradient_variance_per_param", "gradient_noise_scale"):
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for split in ("probe", "old_data"):
            xs, ys = [], []
            for s, r in zip(steps, rows):
                v = r.get(split, {}).get("gradient", {}).get(grad_key)
                if v is not None:
                    xs.append(s)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, marker="o", label=split)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("global_step")
        ax.set_ylabel(grad_key)
        ax.set_title(grad_key)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if log_y:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{grad_key}.png"), dpi=120)
        plt.close(fig)

    # ---------- dead units overall (per split) ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for split in ("probe", "old_data"):
        xs, ys = [], []
        for s, r in zip(steps, rows):
            du = r.get(split, {}).get("dead_units", {}).get("_overall")
            if du is not None:
                xs.append(s)
                ys.append(du["dead_fraction"])
        if xs:
            ax.plot(xs, ys, marker="o", label=split)
            plotted = True
    if plotted:
        ax.set_xlabel("global_step")
        ax.set_ylabel("dead unit fraction (overall)")
        ax.set_title("Dead units (overall, across all MLP layers)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "dead_units_overall.png"), dpi=120)
    plt.close(fig)

    # ---------- per-layer dead units, one figure per checkpoint ----------
    for s, r in zip(steps, rows):
        for split in ("probe", "old_data"):
            du = r.get(split, {}).get("dead_units")
            if not du:
                continue
            layers, fracs = [], []
            for k, v in du.items():
                if k == "_overall":
                    continue
                m = k.split("_")
                try:
                    layers.append(int(m[-1]))
                    fracs.append(v["dead_fraction"])
                except Exception:
                    continue
            if not layers:
                continue
            order = sorted(range(len(layers)), key=lambda i: layers[i])
            layers = [layers[i] for i in order]
            fracs = [fracs[i] for i in order]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(layers, fracs)
            ax.set_xlabel("layer index")
            ax.set_ylabel("dead unit fraction")
            ax.set_title(f"Dead units per layer ({split}) — step {s}")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"dead_units_per_layer_{split}_step_{s}.png"),
                dpi=120,
            )
            plt.close(fig)

    print(f"Wrote checkpoint plots to {out_dir}")


def main():
    args = parse_args()
    plot_train(args.run_dir, args.log_y)
    plot_checkpoints(args.run_dir, args.log_y)


if __name__ == "__main__":
    main()
