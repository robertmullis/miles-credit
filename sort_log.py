#!/usr/bin/env python3
"""Print one epoch's metrics from a CREDIT training_log.csv: the totals 
(loss, RMSE, r2), then a per-variable table (RMSE, r2, bias) sorted by 
r2, RMSE or name.

    sort_log.py path/to/training_log.csv
    sort_log.py path/to/training_log.csv --sort rmse --epoch 20
"""

import argparse
import csv
import os
import sys

def val(row, col):
    """Return row[col] as a float, or None if the column is missing or blank."""

    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None

def short(var):
    """Turn a column key like 'CESM/prognostic/3d/U' into 'U (3d)'."""

    parts = var.split("/")
    return f"{parts[-1]} ({parts[-2]})" if len(parts) >= 2 else var

def collect(last, split):
    """Return one dict per variable (rmse, r2, bias) for the given split ('train' or 'valid').
    Variables are found from the '<split>_rmse/<var>' columns"""

    prefix = f"{split}_rmse/"
    records = []
    for col in last:
        if not col.startswith(prefix):
            continue
        var = col[len(prefix):]
        rmse = val(last, col)
        if rmse is None: # blank cell, variable not logged
            continue
        records.append(
            {
                "variable": var,
                "rmse": rmse,
                "r2": val(last, f"{split}_r2score/{var}"),
                "bias": val(last, f"{split}_bias/{var}"),
            }
        )
    return records

def render_totals(split, row):
    """Print the split's overall loss, RMSE, and r2."""

    loss = val(row, f"{split}_loss")
    rmse = val(row, f"{split}_rmse")
    r2 = val(row, f"{split}_r2score")
    fmt = lambda x, spec: format(x, spec) if x is not None else "--" # show '--' for missing values
    print(f"{'TOTAL':<20}{'loss':>12}{'RMSE':>12}{'r2':>9}")
    print(f"{'':<20}{fmt(loss, '12.5f'):>12}{fmt(rmse, '12.4g'):>12}{fmt(r2, '9.4f'):>9}")
    print()

def render(split, records, epoch, sort, row):
    """Print the totals, then the per-variable table sorted by `sort`."""

    print(f"\n{split} metrics at epoch {epoch}, sorted by {sort}\n")
    render_totals(split, row)
    if not records:
        print("  (no data)\n")
        return

    keys = { # r2: highest first; rmse: smallest first; name: alphabetical
        "r2": lambda r: -(r["r2"] if r["r2"] is not None else -9e9),
        "rmse": lambda r: r["rmse"],
        "name": lambda r: r["variable"],
    }
    records.sort(key=keys[sort])

    print(f"{'Variable':<20}{'RMSE':>12}{'r2':>9}{'bias':>13}")
    print("-" * 54)
    for r in records:
        r2_text = f"{r['r2']:9.3f}" if r["r2"] is not None else f"{'--':>9}"
        bias_text = f"{r['bias']:13.4g}" if r["bias"] is not None else f"{'--':>13}"
        print(f"{short(r['variable']):<20}{r['rmse']:12.4g}{r2_text}{bias_text}")
    print()

def pick(rows, wanted):
    """Return the row for epoch `wanted`, or the last row if None."""

    if wanted is None:
        return rows[-1]
    matches = [r for r in rows if val(r, "epoch") == wanted]
    if not matches:
        have = sorted({int(e) for e in (val(r, "epoch") for r in rows) if e is not None})
        sys.exit(f"epoch {wanted} not in log. Available: {have}")
    return matches[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to training_log.csv")
    ap.add_argument("--sort", default="r2", choices=["r2", "rmse", "name"], 
                    help="column to sort the per-variable table by (default: %(default)s)")
    ap.add_argument("--epoch", type=int, default=None, help="epoch to report (default: last row)")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        sys.exit(f"no such file: {args.log}")
    with open(args.log, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{args.log} has no data rows")

    row = pick(rows, args.epoch)
    train = collect(row, "train")
    valid = collect(row, "valid")
    if not train and not valid:
        # show which metric prefixes the log does have
        found = sorted({c.split("/")[0] for c in row if "/" in c})
        sys.exit(f"no 'train_rmse/*' or 'valid_rmse/*' columns. Found: {found}")

    epoch = row.get("epoch", rows.index(row))
    render("train", train, epoch, args.sort, row)
    render("valid", valid, epoch, args.sort, row)

if __name__ == "__main__":
    main()
