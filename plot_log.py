#!/usr/bin/env python3
"""
Plot the r2 training curves from a CREDIT training_log.csv: training or 
validation; per-variable or total. Save as a PNG.

    plot_log.py path/to/training_log.csv
    plot_log.py path/to/training_log.csv --split valid --total
"""

import argparse
import csv
import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def val(row, col):
    """Return row[col] as a float, or None if column is missing or blank."""

    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


def short(var):
    """Turn a column key like 'CESM/prognostic/3d/U' into 'U (3d)'."""

    parts = var.split("/")
    return f"{parts[-1]} ({parts[-2]})" if len(parts) >= 2 else var


METRIC = "r2score" # metric for curve (epochs vs r2)


def series(rows, split):
    """Return each variable's r2 points per epoch."""

    prefix = f"{split}_{METRIC}/"
    names = [c[len(prefix):] for c in rows[0] if c.startswith(prefix)]
    out = {}
    for var in names:
        pts = [(val(r, "epoch"), val(r, prefix + var)) for r in rows]
        pts = [(e, v) for e, v in pts if e is not None and v is not None]
        if pts:
            out[var] = pts # e.g. out["CESM/prognostic/3d/T"] = [(0, 0.91), (1, 0.95), ...]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to training_log.csv")
    ap.add_argument("--split", default="train", choices=["train", "valid"], help="which split to plot")
    ap.add_argument("--vars", nargs="+", default=None, help="variable short names e.g. PRECT T")
    ap.add_argument("--ymin", type=float, default=None, help="minimum y of plot (default: auto)")
    ap.add_argument("--ymax", type=float, default=1, help="maximum y of plot (default: %(default)s)")
    ap.add_argument("--total", action="store_true", help="plot the training total (r2 or loss)")
    ap.add_argument("--loss", action="store_true", help="plot total loss (must use --total)")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.log):
        sys.exit(f"no such file: {args.log}")
    with open(args.log, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{args.log} has no data rows")

    seen = {}

    # keep one row per epoch, sorted by epoch
    for r in rows:
        e = val(r, "epoch")
        if e is not None:
            seen[e] = r
    rows = [seen[k] for k in sorted(seen)]

    if args.loss and not args.total:
        sys.exit("can not use --loss without --total")

    # --total: plot the split's overall r2 per epoch
    if args.total:
        if args.vars:
            sys.exit("can not use --total and --vars together")
        if args.loss:
            col = f"{args.split}_loss"
            pts = [(val(r,"epoch"), val(r,col)) for r in rows]
            pts = [(e,v) for e, v in pts if e is not None and v is not None]
            if not pts:
                sys.exit(f"no '{col}' column in {args.log}")
            data = {"total": pts}
        else:
            col = f"{args.split}_{METRIC}"
            pts = [(val(r,"epoch"), val(r,col)) for r in rows]
            pts = [(e,v) for e, v in pts if e is not None and v is not None]
            if not pts:
                sys.exit(f"no '{col}' column in {args.log}")
            data = {"total": pts}

    # save 'data' as a dictionary of each variable and its corresponding r2 points
    else:
        data = series(rows, args.split)
        if not data:
            sys.exit(f"no '{args.split}_{METRIC}/*' columns in {args.log}")
        if args.vars:
            available = {k.split("/")[-1].upper(): k for k in data}
            wanted = [v.upper() for v in args.vars]
            unknown = [v for v, w in zip(args.vars, wanted) if w not in available]
            if unknown:
                sys.exit(f"unknown variable(s): {', '.join(unknown)}\navailable: {', '.join(sorted(available))}")
            data = {available[w]: data[available[w]] for w in wanted}

    order = sorted(data, key=lambda k: -data[k][-1][1]) # sort data by final r2 score
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, var in enumerate(order):
        xs = [p[0] for p in data[var]]
        ys = [p[1] for p in data[var]]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.4, color=cmap(i % 20), label=short(var))

    if args.ymin is not None:
        ax.set_ylim(bottom=args.ymin)
    if args.ymax is not None and not args.loss:
        ax.set_ylim(top=args.ymax)
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{args.split}_{METRIC}")
    ax.set_title(f"{args.split} {METRIC} by epoch — {os.path.basename(os.path.dirname(args.log))}")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()

    run = os.path.basename(os.path.dirname(args.log))
    # save plot to args.out or default path
    out = args.out or os.path.expandvars(
        f"/glade/work/$USER/miles-credit/camulator-plots/{args.split}-plots/{args.split}_{run}.png"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  ({len(order)} variables, epochs {int(min(xs))}-{int(max(xs))})")


if __name__ == "__main__":
    main()
