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

# Shared plot style (serif, larger fonts) — matches the Section 4.2 benchmark plots.
PLOT_STYLE = {
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Liberation Serif", "Times", "serif"],
    "font.size":        18, "axes.titlesize":  20, "axes.labelsize":  18,
    "xtick.labelsize":  14, "ytick.labelsize":  14, "legend.fontsize": 14,
    "axes.linewidth":   1.4, "grid.linewidth":   0.9, "lines.linewidth": 2.0,
}
plt.rcParams.update(PLOT_STYLE)


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
    p.add_argument("--run_dir", type=str, default=None,
                   help="Single run dir (legacy single-run mode).")
    p.add_argument("--run_dirs", type=str, default=None,
                   help="Comma-separated list of run dirs to overlay on the same plots.")
    p.add_argument("--labels", type=str, default=None,
                   help="Comma-separated labels (one per run_dir). Defaults to basename of each run_dir.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write comparison plots (compare mode). Defaults to "
                        "<first_run_dir>/../plots_compare_<tag>.")
    p.add_argument("--tag", type=str, default="compare",
                   help="Suffix for the default output_dir in compare mode.")
    p.add_argument("--log_y", action="store_true",
                   help="Use log scale for grad norms / KL plots.")
    p.add_argument("--train_skip_steps", type=int, default=0,
                   help="Drop the first N training steps from training-curve plots so the y-axis "
                        "isn't dominated by initial spikes (e.g. step-0 loss/entropy).")
    p.add_argument("--y_quantile_clip", type=float, default=0.0,
                   help="If > 0 and < 0.5, clip the y-axis of training plots to "
                        "[q, 1-q] quantiles of the data so high-variance early steps don't \"blow up\" "
                        "the visible range. Suggest 0.02 (2%%/98%%).")
    return p.parse_args()


_PRETTY = {
    "kl_from_init": "KL(p_init || p_curr)",
    "kl_from_previous_checkpoint": "KL(p_prev || p_curr)",
}


def _pretty(key: str) -> str:
    return _PRETTY.get(key, key)


def _apply_y_clip(ax, vals_iter, q: float):
    """Clip y-axis to [q, 1-q] quantiles of the supplied values, with 5%% padding."""
    import numpy as np
    if q <= 0 or q >= 0.5:
        return
    arr = np.asarray([v for v in vals_iter if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    lo = float(np.quantile(arr, q))
    hi = float(np.quantile(arr, 1.0 - q))
    if not (hi > lo):
        return
    pad = 0.05 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def plot_train(run_dir: str, log_y: bool, train_skip_steps: int = 0,
               y_quantile_clip: float = 0.0):
    metrics_path = os.path.join(run_dir, "metrics", "train_metrics.jsonl")
    if not os.path.exists(metrics_path):
        print(f"No {metrics_path}; skipping training plots.")
        return
    rows = read_jsonl(metrics_path)
    if not rows:
        return
    if train_skip_steps > 0:
        rows = [r for r in rows if r.get("global_step", 0) >= train_skip_steps]
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
        ax.set_ylabel(_pretty(key))
        ax.set_title(_pretty(key))
        ax.grid(True, alpha=0.3)
        if log_y and ("grad" in key or "kl" in key or "update_norm" == key):
            ax.set_yscale("log")
        else:
            _apply_y_clip(ax, vals, y_quantile_clip)
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
        for split in ("probe", "old_data", "new_data"):
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
        ax.set_ylabel(_pretty(key))
        ax.set_title(_pretty(key))
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
        for split in ("probe", "old_data", "new_data"):
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
    for split in ("probe", "old_data", "new_data"):
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
        for split in ("probe", "old_data", "new_data"):
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


# ---------------------------- compare mode --------------------------------

def _load_train_rows(run_dir: str):
    path = os.path.join(run_dir, "metrics", "train_metrics.jsonl")
    return read_jsonl(path) if os.path.exists(path) else []


def _load_ckpt_rows(run_dir: str):
    files = sorted(
        glob(os.path.join(run_dir, "checkpoints", "step_*", "eval_metrics.json")),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]),
    )
    return [json.load(open(p)) for p in files]


def _grid(n: int, ncols: int = 3):
    """Make a (nrows x ncols) subplot grid sized for n panels; return (fig, flat_axes)."""
    ncols = min(ncols, max(1, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.4 * nrows),
                             squeeze=False)
    return fig, list(axes.flatten())


def _shared_legend(fig, axes, n_used):
    """Attach one figure-level legend, taken from whichever panel has the most series."""
    handles, labs = [], []
    for ax in axes[:n_used]:
        h, l = ax.get_legend_handles_labels()
        if len(l) > len(labs):
            handles, labs = h, l
    if handles:
        fig.legend(handles, labs, loc="lower center",
                   ncol=min(len(labs), 4))


def plot_summary_train(train_rows_per_run, labels, colors, out_path: str,
                       log_y: bool, y_quantile_clip: float = 0.0):
    """All training metrics as subplots in a single figure (runs overlaid)."""
    keys = [k for k in TRAIN_METRICS
            if any(any(r.get(k) is not None for r in rows) for rows in train_rows_per_run)]
    if not keys:
        return
    fig, axes = _grid(len(keys))
    for idx, key in enumerate(keys):
        ax = axes[idx]
        all_vals = []
        for rows, lbl, c in zip(train_rows_per_run, labels, colors):
            if not rows:
                continue
            xs, ys = [], []
            for r in rows:
                v = r.get(key)
                if v is not None:
                    xs.append(r["global_step"])
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, lw=1.0, label=lbl, color=c)
                all_vals.extend(ys)
        ax.set_title(_pretty(key), fontweight="bold")
        ax.set_xlabel("global_step")
        ax.grid(True, alpha=0.3)
        if log_y and ("grad" in key or "kl" in key or key == "update_norm"):
            ax.set_yscale("log")
        else:
            _apply_y_clip(ax, all_vals, y_quantile_clip)
    for ax in axes[len(keys):]:
        ax.axis("off")
    _shared_legend(fig, axes, len(keys))
    fig.suptitle("Training metrics", fontsize=24, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote training summary to {out_path}")


def plot_summary_checkpoints(ckpt_rows_per_run, labels, colors, out_path: str,
                             log_y: bool):
    """All checkpoint metrics as subplots, overlaying run x (old_data/new_data/probe)."""
    splits = ("probe", "old_data", "new_data")
    linestyles = {"probe": "--", "old_data": "-", "new_data": ":"}

    # (title, getter(row, split) -> value or None, prefers_log_scale)
    panels = []
    for key in CKPT_SCALAR_FIELDS:
        panels.append((_pretty(key),
                       (lambda r, s, key=key: r.get(s, {}).get(key)),
                       ("kl" in key)))
    for gk in ("mean_gradient_norm", "gradient_variance_per_param", "gradient_noise_scale"):
        panels.append((gk,
                       (lambda r, s, gk=gk: r.get(s, {}).get("gradient", {}).get(gk)),
                       True))

    def _dead(r, s):
        du = r.get(s, {}).get("dead_units", {}).get("_overall")
        return du["dead_fraction"] if du else None
    panels.append(("dead unit fraction (overall)", _dead, False))

    def _has_data(getter):
        return any(getter(r, s) is not None
                   for rows in ckpt_rows_per_run for r in rows for s in splits)
    panels = [p for p in panels if _has_data(p[1])]
    if not panels:
        return

    fig, axes = _grid(len(panels))
    for idx, (title, getter, prefers_log) in enumerate(panels):
        ax = axes[idx]
        for rows, lbl, c in zip(ckpt_rows_per_run, labels, colors):
            for s in splits:
                xs, ys = [], []
                for r in rows:
                    v = getter(r, s)
                    if v is not None:
                        xs.append(r["global_step"])
                        ys.append(v)
                if xs:
                    ax.plot(xs, ys, marker="o", ms=3, linestyle=linestyles[s],
                            color=c, label=f"{lbl}/{s}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("global_step")
        ax.grid(True, alpha=0.3)
        if log_y and prefers_log:
            ax.set_yscale("log")
    for ax in axes[len(panels):]:
        ax.axis("off")
    _shared_legend(fig, axes, len(panels))
    fig.suptitle("Comparison on old vs new data", fontsize=24, fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote checkpoint comparison summary to {out_path}")


def plot_compare(run_dirs: list[str], labels: list[str], out_dir: str, log_y: bool,
                 train_skip_steps: int = 0, y_quantile_clip: float = 0.0):
    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(out_dir, "train")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(run_dirs))]

    # ---------- training curves ----------
    train_rows_per_run = [_load_train_rows(r) for r in run_dirs]
    if train_skip_steps > 0:
        train_rows_per_run = [
            [r for r in rows if r.get("global_step", 0) >= train_skip_steps]
            for rows in train_rows_per_run
        ]
    for key in TRAIN_METRICS:
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        all_vals = []
        for rows, lbl, c in zip(train_rows_per_run, labels, colors):
            if not rows:
                continue
            xs, ys = [], []
            for r in rows:
                v = r.get(key)
                if v is not None:
                    xs.append(r["global_step"])
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, lw=1.0, label=lbl, color=c)
                all_vals.extend(ys)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("global_step")
        ax.set_ylabel(_pretty(key))
        ax.set_title(_pretty(key))
        ax.grid(True, alpha=0.3)
        ax.legend()
        if log_y and ("grad" in key or "kl" in key or key == "update_norm"):
            ax.set_yscale("log")
        else:
            _apply_y_clip(ax, all_vals, y_quantile_clip)
        fig.tight_layout()
        fig.savefig(os.path.join(train_dir, f"{key}.png"), dpi=120)
        plt.close(fig)

    # ---------- checkpoint scalars ----------
    ckpt_rows_per_run = [_load_ckpt_rows(r) for r in run_dirs]
    splits = ("probe", "old_data", "new_data")
    linestyles = {"probe": "--", "old_data": "-", "new_data": ":"}

    for key in CKPT_SCALAR_FIELDS:
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for rows, lbl, c in zip(ckpt_rows_per_run, labels, colors):
            for split in splits:
                xs, ys = [], []
                for r in rows:
                    v = r.get(split, {}).get(key)
                    if v is not None:
                        xs.append(r["global_step"])
                        ys.append(v)
                if xs:
                    ax.plot(xs, ys, marker="o", linestyle=linestyles[split],
                            color=c, label=f"{lbl}/{split}")
                    plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("global_step")
        ax.set_ylabel(_pretty(key))
        ax.set_title(_pretty(key))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        if log_y and "kl" in key:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, f"{key}.png"), dpi=120)
        plt.close(fig)

    # ---------- gradient diagnostics ----------
    for grad_key in ("mean_gradient_norm", "gradient_variance_per_param", "gradient_noise_scale"):
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for rows, lbl, c in zip(ckpt_rows_per_run, labels, colors):
            for split in splits:
                xs, ys = [], []
                for r in rows:
                    v = r.get(split, {}).get("gradient", {}).get(grad_key)
                    if v is not None:
                        xs.append(r["global_step"])
                        ys.append(v)
                if xs:
                    ax.plot(xs, ys, marker="o", linestyle=linestyles[split],
                            color=c, label=f"{lbl}/{split}")
                    plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("global_step")
        ax.set_ylabel(grad_key)
        ax.set_title(grad_key)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        if log_y:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, f"{grad_key}.png"), dpi=120)
        plt.close(fig)

    # ---------- dead units overall ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for rows, lbl, c in zip(ckpt_rows_per_run, labels, colors):
        for split in splits:
            xs, ys = [], []
            for r in rows:
                du = r.get(split, {}).get("dead_units", {}).get("_overall")
                if du is not None:
                    xs.append(r["global_step"])
                    ys.append(du["dead_fraction"])
            if xs:
                ax.plot(xs, ys, marker="o", linestyle=linestyles[split],
                        color=c, label=f"{lbl}/{split}")
                plotted = True
    if plotted:
        ax.set_xlabel("global_step")
        ax.set_ylabel("dead unit fraction (overall)")
        ax.set_title("Dead units (overall, across all MLP layers)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, "dead_units_overall.png"), dpi=120)
    plt.close(fig)

    # ---------- single-figure summaries ----------
    plot_summary_train(train_rows_per_run, labels, colors,
                       os.path.join(out_dir, "summary_training_metrics.png"),
                       log_y, y_quantile_clip=y_quantile_clip)
    plot_summary_checkpoints(ckpt_rows_per_run, labels, colors,
                             os.path.join(out_dir, "comparison_old_new_data.png"),
                             log_y)

    print(f"Wrote comparison plots to {out_dir}")
    print(f"  runs: {list(zip(labels, run_dirs))}")


def main():
    args = parse_args()
    if args.run_dirs:
        run_dirs = [r.strip() for r in args.run_dirs.split(",") if r.strip()]
        if args.labels:
            labels = [s.strip() for s in args.labels.split(",")]
            assert len(labels) == len(run_dirs), "--labels count must match --run_dirs"
        else:
            labels = [os.path.basename(os.path.normpath(r)) for r in run_dirs]
        if args.output_dir:
            out_dir = args.output_dir
        else:
            parent = os.path.dirname(os.path.normpath(run_dirs[0]))
            out_dir = os.path.join(parent, f"plots_{args.tag}")
        plot_compare(run_dirs, labels, out_dir, args.log_y,
                     train_skip_steps=args.train_skip_steps,
                     y_quantile_clip=args.y_quantile_clip)
        return
    if not args.run_dir:
        raise SystemExit("Provide either --run_dir or --run_dirs")
    plot_train(args.run_dir, args.log_y,
               train_skip_steps=args.train_skip_steps,
               y_quantile_clip=args.y_quantile_clip)
    plot_checkpoints(args.run_dir, args.log_y)

    # single-figure summaries (single-run = one overlaid "run")
    label = os.path.basename(os.path.normpath(args.run_dir))
    color = [plt.get_cmap("tab10")(0)]
    plots_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    train_rows = _load_train_rows(args.run_dir)
    if args.train_skip_steps > 0:
        train_rows = [r for r in train_rows
                      if r.get("global_step", 0) >= args.train_skip_steps]
    plot_summary_train([train_rows], [label], color,
                       os.path.join(plots_dir, "summary_training_metrics.png"),
                       args.log_y, y_quantile_clip=args.y_quantile_clip)
    plot_summary_checkpoints([_load_ckpt_rows(args.run_dir)], [label], color,
                             os.path.join(plots_dir, "comparison_old_new_data.png"),
                             args.log_y)


if __name__ == "__main__":
    main()
