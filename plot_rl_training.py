#!/usr/bin/env python3
"""
Plot RL (verl) training-time metrics logged via MLflow for a single run.

Reads:
  {exp_path}/mlflow/<experiment_id>/<run_id>/metrics/<metric_path>

Each metric file is whitespace-separated: `<timestamp_ms> <value> <step>`.

Plots one figure with subplots for:
  - critic/rewards/mean   (training reward)
  - critic/advantages/mean (advantages)
  - actor/kl_loss
  - actor/entropy
  - actor/grad_norm

Usage:
  python plot_rl_training.py \
      --exp_path /home/t-jinshen/amlt/qwen3b_cd_noformat/qwen3b_cd_noformat/qwen3b_cd_noformat
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_STYLE = {
    "font.family":     "serif",
    "font.serif":      ["DejaVu Serif", "Liberation Serif", "Times", "serif"],
    "font.size":       16, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    "axes.linewidth":  1.2, "grid.linewidth":  0.8, "lines.linewidth": 1.8,
}
plt.rcParams.update(PLOT_STYLE)


# (metric_path_on_disk, panel_title, y_label)
PANELS = [
    ("critic/rewards/mean",     "Training Reward (critic/rewards/mean)", "reward"),
    ("critic/advantages/mean",  "Advantages (critic/advantages/mean)",   "advantage"),
    ("actor/kl_loss",           "KL Loss (actor/kl_loss)",               "kl_loss"),
    ("actor/entropy",           "Entropy (actor/entropy)",               "entropy"),
    ("actor/grad_norm",         "Grad Norm (actor/grad_norm)",           "grad_norm"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_path", required=True,
                   help="Experiment root that contains the `mlflow/` directory.")
    p.add_argument("--output_dir", default=None,
                   help="Where to save plots. Default: {exp_path}/plots_training.")
    p.add_argument("--label", default=None,
                   help="Legend label for the run. Default: basename of --exp_path.")
    p.add_argument("--smooth", type=int, default=1,
                   help="Centered moving-average window for the smoothed line. "
                        "Set to 1 to disable smoothing (default: 1).")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Optional upper bound on global_step.")
    p.add_argument("--log_y_grad_norm", action="store_true",
                   help="Use log y-scale for the grad_norm panel.")
    return p.parse_args()


# ----------------------------- IO helpers ---------------------------------

def _read_metric(file_path: str) -> tuple[list[int], list[float]]:
    """Read an MLflow metric file: lines of `<ts_ms> <value> <step>`."""
    steps, values = [], []
    if not os.path.isfile(file_path):
        return steps, values
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                step = int(parts[2])
                val = float(parts[1])
            except ValueError:
                continue
            steps.append(step)
            values.append(val)
    return steps, values


def _discover_runs(exp_path: str) -> list[str]:
    """Return list of run dirs under {exp_path}/mlflow/<exp_id>/<run_id>/."""
    mlflow_root = os.path.join(exp_path, "mlflow")
    if not os.path.isdir(mlflow_root):
        return []
    exp_dirs = sorted(
        d for d in os.listdir(mlflow_root)
        if os.path.isdir(os.path.join(mlflow_root, d)) and d.isdigit()
    )
    runs: list[str] = []
    for exp_id in exp_dirs:
        exp_id_path = os.path.join(mlflow_root, exp_id)
        for item in sorted(os.listdir(exp_id_path)):
            full = os.path.join(exp_id_path, item)
            # MLflow run ids are 32-char hex; skip meta.yaml etc.
            if os.path.isdir(full) and len(item) == 32 and not item.startswith("."):
                runs.append(full)
    return runs


def _load_metric_across_runs(exp_path: str, metric_rel_path: str,
                             max_steps: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate a metric across all runs found under exp_path; dedupe by step
    keeping the last value seen for each step."""
    by_step: dict[int, float] = {}
    for run in _discover_runs(exp_path):
        f = os.path.join(run, "metrics", metric_rel_path)
        s, v = _read_metric(f)
        for step, val in zip(s, v):
            if max_steps is not None and step > max_steps:
                continue
            by_step[step] = val
    if not by_step:
        return np.array([]), np.array([])
    steps = np.array(sorted(by_step.keys()))
    values = np.array([by_step[s] for s in steps], dtype=float)
    return steps, values


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1 or y.size == 0:
        return y
    window = min(window, y.size)
    kernel = np.ones(window, dtype=float) / float(window)
    # 'same' keeps the array length; edges are slightly biased toward the mean.
    return np.convolve(y, kernel, mode="same")


# ----------------------------- plotting -----------------------------------

def _grid(n: int, ncols: int = 3):
    ncols = min(ncols, max(1, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.4 * nrows),
                             squeeze=False)
    return fig, list(axes.flatten())


def main():
    args = parse_args()
    exp_path = os.path.abspath(args.exp_path)
    label = args.label or os.path.basename(os.path.normpath(exp_path))
    out_dir = args.output_dir or os.path.join(exp_path, "plots_training")
    os.makedirs(out_dir, exist_ok=True)

    runs = _discover_runs(exp_path)
    if not runs:
        raise SystemExit(f"No MLflow runs found under {os.path.join(exp_path, 'mlflow')}")
    print(f"Found {len(runs)} MLflow run(s) under {exp_path}")

    # ---------- combined figure ----------
    fig, axes = _grid(len(PANELS), ncols=3)
    color = "C0"

    for ax, (metric_path, title, ylabel) in zip(axes, PANELS):
        steps, values = _load_metric_across_runs(exp_path, metric_path, args.max_steps)
        if steps.size == 0:
            ax.set_title(f"{title}\n(no data)")
            ax.grid(True, alpha=0.3)
            continue

        # Raw points (faint) + smoothed line (bold) when smoothing is on.
        if args.smooth and args.smooth > 1:
            ax.plot(steps, values, color=color, alpha=0.25, linewidth=1.0,
                    label=f"{label} (raw)")
            smoothed = _moving_average(values, args.smooth)
            ax.plot(steps, smoothed, color=color, linewidth=2.0,
                    label=f"{label} (ma={args.smooth})")
        else:
            ax.plot(steps, values, color=color, linewidth=1.8, label=label)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("global_step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if args.log_y_grad_norm and metric_path.endswith("grad_norm"):
            ax.set_yscale("log")
        ax.legend(fontsize=10)

        # Also save individual panel.
        fig_i, ax_i = plt.subplots(figsize=(8, 4.8))
        if args.smooth and args.smooth > 1:
            ax_i.plot(steps, values, color=color, alpha=0.25, linewidth=1.0,
                      label=f"{label} (raw)")
            ax_i.plot(steps, _moving_average(values, args.smooth),
                      color=color, linewidth=2.0,
                      label=f"{label} (ma={args.smooth})")
        else:
            ax_i.plot(steps, values, color=color, linewidth=1.8, label=label)
        ax_i.set_title(title, fontweight="bold")
        ax_i.set_xlabel("global_step")
        ax_i.set_ylabel(ylabel)
        ax_i.grid(True, alpha=0.3)
        if args.log_y_grad_norm and metric_path.endswith("grad_norm"):
            ax_i.set_yscale("log")
        ax_i.legend(fontsize=11)
        fig_i.tight_layout()
        fname = metric_path.replace("/", "_") + ".png"
        fig_i.savefig(os.path.join(out_dir, fname), dpi=120)
        plt.close(fig_i)

    # Hide any unused axes in the grid.
    for ax in axes[len(PANELS):]:
        ax.axis("off")

    fig.suptitle(f"RL Training Metrics — {label}", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    summary_path = os.path.join(out_dir, "summary_training.png")
    fig.savefig(summary_path, dpi=120)
    plt.close(fig)

    print(f"Wrote plots to {out_dir}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
