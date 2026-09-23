"""Print a size-saved and accuracy comparison of two CREDIT runs. Used to measure the effect of
BitRound compression on file size and validation accuracy. Print total and variable-by-variable
analysis of two runs. By default the baseline is the camulator_gen2 eval log.

    compare_runs.py -c path/to/config.yml -l path/to/training_log.csv
    compare_runs.py -c path/to/config.yml -l path/to/training_log.csv --per-var
    compare_runs.py -c path/to/config.yml -l path/to/training_log.csv -b path/to/baseline/training_log.csv
"""

import argparse, csv, os, sys
from bitround_estimate_size import estimate_size_saved

# default baseline: eval log of the unrounded camulator_gen2 run
BASELINE = "/glade/derecho/scratch/mullis/miles-credit/CREDIT_runs/camulator_gen2_eval/training_log.csv"

def read_row(path, epoch=None):
    """Return the row for `epoch`, or the last row if epoch is None."""

    with open(path,newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} has no data rows")
    if epoch is None:
        return rows[-1] # return last row
    for r in rows:
        if int(r["epoch"]) == epoch:
            return r
    sys.exit(f"epoch {epoch} not found in {path}")

def per_var(row,split,metric):
    """Return {variable name: value} for every column starting with '<split>_<metric>/'."""

    prefix = f"{split}_{metric}/"
    out = {}
    for col, var in row.items():
        if col.startswith(prefix):
            try:
                out[col.split("/")[-1]] = float(var)
            except(TypeError, ValueError):
                pass
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to config with BitRound Preblock")
    ap.add_argument("-l", "--log", required=True, help ="evaluation training log of rounded run")
    ap.add_argument("-b", "--baseline", default=BASELINE, help="evaluation training log of baseline run")
    ap.add_argument("-bc", "--baseline-config", default=None, help="config of baseline run")
    ap.add_argument("--epoch", type=int, default=None, help="epoch to report (default: last row)")
    ap.add_argument("--baseline-epoch", type=int, default=None, help="baseline epoch to report (default: last row)")
    ap.add_argument("--split", default="valid", choices=["train", "valid"], help="'train' or 'valid'")
    ap.add_argument("--per-var", action="store_true", help="print a variable-by-variable table")
    ap.add_argument("-n", "--n-samples", type=int, default=24)
    args = ap.parse_args()

    for p in (args.config, args.log, args.baseline, args.baseline_config):
        if p is None:
            continue
        if not os.path.exists(p):
            sys.exit(f"no such file: {p}")
    
    if args.baseline_config and args.baseline == BASELINE:
        sys.exit("--baseline-config needs -b pointing at that run's eval log")
    
    base = read_row(args.baseline, args.baseline_epoch) # baseline row (default: last)
    rounded = read_row(args.log, args.epoch) # bitround row (default: last)
    
    # GB/yr uncompressed, lossless, and bitrounded for this config
    rows, tot = estimate_size_saved(args.config, args.n_samples)
    round_gb = tot[2] / 1e9 # bytes/yr -> GB/yr

    # estimate baseline size if -bc is passed
    if args.baseline_config:
        b_rows, b_tot = estimate_size_saved(args.baseline_config, args.n_samples)
        orig_gb = b_tot[2] / 1e9
        b_label = os.path.splitext(os.path.basename(args.baseline_config))[0]
    # else compare against the current archive (lossless compression only)
    else:
        b_rows = None
        orig_gb = tot[1] / 1e9
        b_label = "original"
    r_label = os.path.splitext(os.path.basename(args.config))[0]

    b_loss = float(base[f"{args.split}_loss"]) # baseline loss
    r_loss = float(rounded[f"{args.split}_loss"]) # bitround loss

    b_r2 = float(base[f"{args.split}_r2score"]) # baseline r2 score
    r_r2 = float(rounded[f"{args.split}_r2score"]) # bitround r2 score

    w = max(len(b_label), len(r_label)) + 2
    print(f"{'':{w}s}{'GB/yr':>8}{args.split + '_loss':>12}{args.split + '_r2':>10}")
    print(f"{b_label:{w}s}{orig_gb:8.2f}{b_loss:12.5f}{b_r2:10.5f}")
    print(f"{r_label:{w}s}{round_gb:8.2f}{r_loss:12.5f}{r_r2:10.5f}")
    print(f"\nsize {100 * (round_gb / orig_gb - 1):+.1f}%, "
          f"loss {100 * (r_loss / b_loss - 1):+.1f}%, "
          f"r2 {r_r2 - b_r2:+.5f}")

    # print per-variable comparison summary
    if args.per_var:
        size = {r[0]: r for r in rows}
        b_size = {r[0]: r for r in b_rows} if b_rows else {}
        b_r2_var = per_var(base, args.split, "r2score")
        r_r2_var = per_var(rounded, args.split, "r2score")
        b_rmse_var = per_var(base, args.split, "rmse")
        r_rmse_var = per_var(rounded, args.split, "rmse")

        names = sorted(set(size)|set(b_r2_var), key=lambda n: -size[n][5] if n in size else 0)

        print(f"\n{'var':10s}{'b bits':>7}{'bits':>5}{'saved':>8}{'base r2':>10}{'round r2':>10}{'dr2':>10}{'drmse':>8}")
        for n in names:
            s = size.get(n)
            bs = b_size.get(n)
            br, rr = b_r2_var.get(n), r_r2_var.get(n)
            bm, rm = b_rmse_var.get(n), r_rmse_var.get(n)

            bits  = f"{s[1]:5d}" if s else f"{'-':>5}"
            b_bits = f"{bs[1]:7d}" if bs else f"{'-':>7}"
            saved = f"{s[5]:7.1f}%" if s else f"{'-':>8}"
            base_r2  = f"{br:10.5f}" if br is not None else f"{'-':>10}"
            round_r2 = f"{rr:10.5f}" if rr is not None else f"{'-':>10}"
            dr2   = f"{rr - br:+10.5f}" if br is not None and rr is not None else f"{'-':>10}"
            drmse = f"{100 * (rm / bm - 1):+7.1f}%" if bm and rm is not None else f"{'-':>8}"

            print(f"{n:10s}{b_bits}{bits}{saved}{base_r2}{round_r2}{dr2}{drmse}")