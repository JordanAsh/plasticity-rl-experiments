#!/usr/bin/env python3
"""
Plot checkpoint-time eval metrics for an RL (verl) run, as produced by
`eval_rl_checkpoints.py`.

Reads:
  {ckpt_root}/global_step_*/eval_metrics.json

Writes:
  {ckpt_root}/plots/<metric>.png
  {ckpt_root}/plots/dead_units_per_layer_<split>_step_<N>.png
  {ckpt_root}/plots/summary_pos_neg.png

Compare mode (overlay multiple runs):
  python plot_rl_metrics.py \
      --ckpt_roots .../qwen3b_cd_noformat/.../checkpoints_hf_format,.../qwen3b_kk/.../checkpoints_hf_format \
      --labels cd,kk

Single-run mode:
  python plot_rl_metrics.py --ckpt_root .../qwen3b_cd_noformat/.../checkpoints_hf_format
"""

import argparse
import json
import os
import re
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the style used by plot_metrics.py.
PLOT_STYLE = {
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Liberation Serif", "Times", "serif"],
    "font.size":        18, "axes.titlesize":  20, "axes.labelsize":  18,
    "xtick.labelsize":  14, "ytick.labelsize":  14, "legend.fontsize": 14,
    "axes.linewidth":   1.4, "grid.linewidth":   0.9, "lines.linewidth": 2.0,
}
plt.rcParams.update(PLOT_STYLE)


SCOPES = ("in_batch", "old_data", "new_data")
POLARITIES = ("positive", "negative")
SCOPE_COLORS = {"in_batch": "C0", "old_data": "C1", "new_data": "C2"}
POL_LINESTYLE = {"positive": "-", "negative": "--"}

CKPT_SCALAR_FIELDS = [
    "loss",
    "token_entropy",
    "kl_from_init",
    "kl_to_init",
    "kl_from_previous_checkpoint",
    "kl_to_previous_checkpoint",
]

_PRETTY = {
    "kl_from_init":                "KL(p_init || p_curr)",
    "kl_to_init":                  "KL(p_curr || p_init)",
    "kl_from_previous_checkpoint": "KL(p_prev || p_curr)",
    "kl_to_previous_checkpoint":   "KL(p_curr || p_prev)",
    "loss":                        "loss",
    "token_entropy":               "token entropy",
}


def _pretty(key: str) -> str:
    return _PRETTY.get(key, key)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_root", type=str, default=None,
                   help="Single root containing global_step_*/eval_metrics.json.")
    p.add_argument("--ckpt_roots", type=str, default=None,
                   help="Comma-separated roots to overlay on each plot.")
    p.add_argument("--labels", type=str, default=None,
                   help="Comma-separated labels (one per --ckpt_roots entry).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write plots. Default: {ckpt_root}/plots (single) or "
                        "{first_root}/../plots_<tag> (compare).")
    p.add_argument("--tag", type=str, default="compare")
    p.add_argument("--log_y", action="store_true",
                   help="Use log y-scale for KL / gradient panels.")
    p.add_argument("--per_layer_dead_units", action="store_true",
                   help="Also emit one bar plot per (split, step) showing per-layer dead-unit fractions.")
    return p.parse_args()


# ----------------------------- IO helpers ---------------------------------

def _load_ckpt_rows(ckpt_root: str) -> list[dict]:
    pat = re.compile(r"global_step_(\d+)$")
    files = []
    for d in os.listdir(ckpt_root):
        m = pat.match(d)
        if not m:
            continue
        f = os.path.join(ckpt_root, d, "eval_metrics.json")
        if os.path.exists(f):
            files.append((int(m.group(1)), f))
    files.sort(key=lambda x: x[0])
    return [json.load(open(f)) for _, f in files]


# ----------------------------- single-run plots ---------------------------

def plot_single(ckpt_root: str, out_dir: str, log_y: bool,
                per_layer_dead_units: bool):
    rows = _load_ckpt_rows(ckpt_root)
    if not rows:
        print(f"No eval_metrics.json found under {ckpt_root}")
        return
    os.makedirs(out_dir, exist_ok=True)

    # ---------- subset sizes ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for scope in SCOPES:
        for pol in POLARITIES:
            xs, ys = [], []
            for r in rows:
                blk = r.get(scope, {})
                if not isinstance(blk, dict):
                    continue
                sub = blk.get(pol)
                if isinstance(sub, dict) and "size" in sub:
                    xs.append(r["global_step"])
                    ys.append(sub["size"])
            if xs:
                ax.plot(xs, ys, marker="o",
                        linestyle=POL_LINESTYLE[pol],
                        color=SCOPE_COLORS[scope],
                        label=f"{scope}/{pol}")
    ax.set_xlabel("global_step")
    ax.set_ylabel("num samples (post-cap, post-len-filter)")
    ax.set_title("Subset sizes per (scope, polarity)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "subset_sizes.png"), dpi=120)
    plt.close(fig)

    # ---------- scalar fields ----------
    for key in CKPT_SCALAR_FIELDS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for scope in SCOPES:
            for pol in POLARITIES:
                xs, ys = [], []
                for r in rows:
                    sub = r.get(scope, {}).get(pol)
                    if isinstance(sub, dict) and key in sub:
                        xs.append(r["global_step"])
                        ys.append(sub[key])
                if xs:
                    ax.plot(xs, ys, marker="o",
                            linestyle=POL_LINESTYLE[pol],
                            color=SCOPE_COLORS[scope],
                            label=f"{scope}/{pol}")
                    plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("global_step")
        ax.set_ylabel(_pretty(key))
        ax.set_title(_pretty(key))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, ncol=2)
        if log_y and "kl" in key:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{key}.png"), dpi=120)
        plt.close(fig)

    # ---------- pos - neg gap, per scope ----------
    for key in CKPT_SCALAR_FIELDS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for scope in SCOPES:
            xs, ys = [], []
            for r in rows:
                p = r.get(scope, {}).get("positive", {})
                n = r.get(scope, {}).get("negative", {})
                if isinstance(p, dict) and isinstance(n, dict) and key in p and key in n:
                    xs.append(r["global_step"])
                    ys.append(p[key] - n[key])
            if xs:
                ax.plot(xs, ys, marker="o", color=SCOPE_COLORS[scope], label=scope)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.axhline(0.0, color="grey", lw=1.0, alpha=0.5)
        ax.set_xlabel("global_step")
        ax.set_ylabel(f"Δ {_pretty(key)} (pos - neg)")
        ax.set_title(f"pos - neg gap: {_pretty(key)}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"gap_{key}.png"), dpi=120)
        plt.close(fig)

    # ---------- dead units overall ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for scope in SCOPES:
        for pol in POLARITIES:
            xs, ys = [], []
            for r in rows:
                du = r.get(scope, {}).get(pol, {}).get("dead_units", {}).get("_overall")
                if du is not None:
                    xs.append(r["global_step"])
                    ys.append(du["dead_fraction"])
            if xs:
                ax.plot(xs, ys, marker="o",
                        linestyle=POL_LINESTYLE[pol],
                        color=SCOPE_COLORS[scope],
                        label=f"{scope}/{pol}")
                plotted = True
    if plotted:
        ax.set_xlabel("global_step")
        ax.set_ylabel("dead unit fraction (overall)")
        ax.set_title("Dead units (overall, across all MLP layers)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "dead_units_overall.png"), dpi=120)
    plt.close(fig)

    # ---------- per-layer dead units ----------
    if per_layer_dead_units:
        for r in rows:
            s = r["global_step"]
            for scope in SCOPES:
                for pol in POLARITIES:
                    du = r.get(scope, {}).get(pol, {}).get("dead_units")
                    if not du:
                        continue
                    layers, fracs = [], []
                    for k, v in du.items():
                        if k == "_overall":
                            continue
                        try:
                            layers.append(int(k.split("_")[-1]))
                            fracs.append(v["dead_fraction"])
                        except Exception:
                            continue
                    if not layers:
                        continue
                    order = sorted(range(len(layers)), key=lambda i: layers[i])
                    layers = [layers[i] for i in order]
                    fracs = [fracs[i] for i in order]
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    ax.bar(layers, fracs, color=SCOPE_COLORS[scope])
                    ax.set_xlabel("layer index")
                    ax.set_ylabel("dead unit fraction")
                    ax.set_title(f"Dead units per layer ({scope}/{pol}) — step {s}")
                    ax.grid(True, axis="y", alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(out_dir,
                                     f"dead_units_per_layer_{scope}_{pol}_step_{s}.png"),
                        dpi=120,
                    )
                    plt.close(fig)

    # ---------- summary figure ----------
    _summary_grid(out_dir, [rows], [os.path.basename(os.path.normpath(ckpt_root))],
                  ["C0"], log_y, fname="summary.png")
    print(f"Wrote plots to {out_dir}")


# ----------------------------- compare mode -------------------------------

def _grid(n: int, ncols: int = 3):
    ncols = min(ncols, max(1, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.4 * nrows),
                             squeeze=False)
    return fig, list(axes.flatten())


def _shared_legend(fig, axes, n_used):
    handles, labs = [], []
    for ax in axes[:n_used]:
        h, l = ax.get_legend_handles_labels()
        if len(l) > len(labs):
            handles, labs = h, l
    if handles:
        fig.legend(handles, labs, loc="lower center",
                   ncol=min(len(labs), 4))


def _summary_grid(out_dir, rows_per_run, labels, colors, log_y, fname="summary.png"):
    panels = []
    for key in CKPT_SCALAR_FIELDS:
        panels.append((_pretty(key),
                       (lambda r, sc, p, key=key: r.get(sc, {}).get(p, {}).get(key)),
                       ("kl" in key)))

    def _dead(r, sc, p):
        du = r.get(sc, {}).get(p, {}).get("dead_units", {}).get("_overall")
        return du["dead_fraction"] if du else None
    panels.append(("dead unit fraction (overall)", _dead, False))

    def _has(getter):
        return any(
            getter(r, sc, p) is not None
            for rows in rows_per_run for r in rows
            for sc in SCOPES for p in POLARITIES
        )
    panels = [pp for pp in panels if _has(pp[1])]
    if not panels:
        return

    fig, axes = _grid(len(panels))
    multi_run = len(rows_per_run) > 1
    for idx, (title, getter, prefers_log) in enumerate(panels):
        ax = axes[idx]
        for rows, lbl, c in zip(rows_per_run, labels, colors):
            for sc in SCOPES:
                for p in POLARITIES:
                    xs, ys = [], []
                    for r in rows:
                        v = getter(r, sc, p)
                        if v is not None:
                            xs.append(r["global_step"])
                            ys.append(v)
                    if not xs:
                        continue
                    color = c if multi_run else SCOPE_COLORS[sc]
                    label = (f"{lbl}/{sc}/{p}" if multi_run
                             else f"{sc}/{p}")
                    ax.plot(xs, ys, marker="o", ms=3,
                            linestyle=POL_LINESTYLE[p],
                            color=color, label=label)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("global_step")
        ax.grid(True, alpha=0.3)
        if log_y and prefers_log:
            ax.set_yscale("log")
    for ax in axes[len(panels):]:
        ax.axis("off")
    _shared_legend(fig, axes, len(panels))
    fig.suptitle("Checkpoint metrics: in_batch / old_data / new_data × pos/neg",
                 fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote summary to {out_path}")


def plot_compare(ckpt_roots: list[str], labels: list[str], out_dir: str,
                 log_y: bool):
    os.makedirs(out_dir, exist_ok=True)
    rows_per_run = [_load_ckpt_rows(r) for r in ckpt_roots]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(ckpt_roots))]

    # Per-metric overlay plots: one figure per scope so the legends stay readable.
    for key in CKPT_SCALAR_FIELDS:
        for scope in SCOPES:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            plotted = False
            for rows, lbl, c in zip(rows_per_run, labels, colors):
                for pol in POLARITIES:
                    xs, ys = [], []
                    for r in rows:
                        v = r.get(scope, {}).get(pol, {}).get(key)
                        if v is not None:
                            xs.append(r["global_step"])
                            ys.append(v)
                    if xs:
                        ax.plot(xs, ys, marker="o", linestyle=POL_LINESTYLE[pol],
                                color=c, label=f"{lbl}/{pol}")
                        plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xlabel("global_step")
            ax.set_ylabel(_pretty(key))
            ax.set_title(f"{_pretty(key)} — {scope}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, ncol=2)
            if log_y and "kl" in key:
                ax.set_yscale("log")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{key}__{scope}.png"), dpi=120)
            plt.close(fig)

    _summary_grid(out_dir, rows_per_run, labels, colors, log_y,
                  fname="summary.png")
    print(f"Wrote comparison plots to {out_dir}")
    print(f"  runs: {list(zip(labels, ckpt_roots))}")


def main():
    args = parse_args()
    if args.ckpt_roots:
        roots = [r.strip() for r in args.ckpt_roots.split(",") if r.strip()]
        labels = ([s.strip() for s in args.labels.split(",")] if args.labels
                  else [os.path.basename(os.path.normpath(r)) for r in roots])
        if len(labels) != len(roots):
            raise SystemExit("--labels count must match --ckpt_roots")
        out_dir = args.output_dir or os.path.join(
            os.path.dirname(os.path.normpath(roots[0])),
            f"plots_{args.tag}",
        )
        plot_compare(roots, labels, out_dir, args.log_y)
        return
    if not args.ckpt_root:
        raise SystemExit("Provide either --ckpt_root or --ckpt_roots")
    out_dir = args.output_dir or os.path.join(args.ckpt_root, "plots")
    plot_single(args.ckpt_root, out_dir, args.log_y, args.per_layer_dead_units)


if __name__ == "__main__":
    main()
