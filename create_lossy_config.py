#!/usr/bin/env python3
"""Create a CAMulator configuration with a BitRound preblock from a template file. Also create
the corresponding evaluation configuration file; with no BitRound preblock, a 0.00 learning 
rate, and 1 epoch. If ``tolerance`` is set, keepbits is chosen per variable from the fitted 
scaler: the smallest k whose rounding error std is ``<= tolerance * std``

    create_lossy_config.py
    create_lossy_config.py -t 05 -e 50
"""
import argparse
import os 
import re

BASE = os.path.expandvars("/glade/work/$USER/miles-credit/camulator_configs/lossy")
RUNS = "/glade/derecho/scratch/$USER/miles-credit/CREDIT_runs"
# yml configuration pattern for the bitround preblock
BITROUND = re.compile(r"^ {4}bitround:\n.*?(?=^ {4}scaler:)", re.M | re.S)

def apply(text, values):
    """Return `text` with the first `key: ...` line replaced for each key, keeping its indentation."""
    for key, value in values.items():
        text = re.sub(rf"^(\s*){key}:.*$", rf"\g<1>{key}: {value}", text, count=1, flags=re.M)
    return text

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tolerance", type=float, default=None, help="set keepbits based on error tolerance (default: None); set as percent (-t 05 -> 5%%)")
    ap.add_argument("-k", "--keepbits", type=int, default=4, help="number of bits to keep if tolerance not set (default: %(default)s)")
    ap.add_argument("-ne", "--num_epochs", type=int, default=10, help="number of epochs per derecho job (default: %(default)s)")
    ap.add_argument("-e", "--epochs", type=int, default=30, help="number of epochs for training (default: %(default)s)")
    ap.add_argument("-tc", "--template", default="/glade/work/mullis/miles-credit/camulator_configs/lossy/my_camulator_lossy_tol_05.yml", help="path to build config file from")
    args = ap.parse_args()

    if args.tolerance is None:
        tag = f"{args.keepbits}bits"
        tolerance = "null"
    else:
        tag = "tol_" + format(args.tolerance, "02g").replace(".", "p") # e.g. -t 05 -> tol_05
        tolerance = format(args.tolerance / 100, "g")
    
    text = open(args.template).read()

    # training config: new run with the chosen rounding
    train = apply(text, {
        "save_loc": f"'{RUNS}/camulator_lossy_{tag}'",
        "load_weights": "False",
        "start_epoch": 0,
        "num_epoch": args.num_epochs,
        "epochs": f"&epochs {args.epochs}",
        "tolerance": tolerance,
        "keepbits": args.keepbits
    })

    # eval config: bitround removed, trained weights scored on 400 validation batches
    evaluate = apply(BITROUND.sub("", text), {
        "save_loc": f"'{RUNS}/camulator_lossy_{tag}_eval'",
        "load_weights": "True",
        "learning_rate": 0.0,
        "batches_per_epoch": 1,
        "valid_batches_per_epoch": 400,
        "start_epoch": 0,
        "num_epoch": 1,
        "epochs": "&epochs 1",
    })

    train_cfg = os.path.join(BASE, f"my_camulator_lossy_{tag}.yml") # -t 05 -> my_camulator_lossy_tol_05.yml
    eval_cfg = os.path.join(BASE, f"my_camulator_lossy_{tag}_eval.yml")
    open(train_cfg, "w").write(train)
    open(eval_cfg, "w").write(evaluate)

    print(f"wrote {train_cfg} & {eval_cfg}")