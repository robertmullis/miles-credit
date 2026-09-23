"""Print the estimated size saved from a CREDIT config yml with a BitRound preblock.

    bitround_estimate_size.py -c path/to/config.yml
    bitround_estimate_size.py -c path/to/config.yml -n 12 -y 4
"""
import yaml, numpy as np, zarr
import glob, os, sys
from numcodecs import BitRound
from credit.preblock.bitround import BitRoundTransform
from bridgescaler import load_scaler_dict

def estimate_size_saved(config, n_samples=24, n_years=5, seed=0):
    """Return (rows, tot): per-variable and total GB/yr uncompressed, lossless, and bitrounded."""

    cfg = yaml.safe_load(open(config))
    args = cfg["preblocks"]["per_step"]["bitround"]["args"] # scaler_path, tolerance, keepbits, and variables

    p = next(s["path"] for s in cfg["data"]["source"]["CESM"]["variables"].values()
            if s.get("path", "").endswith("zarr")) # yearly zarr store pattern (contains %Y)
    stores = sorted(glob.glob(os.path.expandvars(p).replace("%Y", "*"))) # every year's store, in year order

    rng = np.random.default_rng(seed)
    picked = rng.choice(stores,size=min(n_years,len(stores)),replace=False) # randomly select n_years from stores
    roots = [zarr.open(s, mode="r") for s in picked]

    # rebuild the preblock to reuse its keepbits logic (_resolve)
    block = BitRoundTransform(
        scaler_path = args["scaler_path"],
        tolerance = args.get("tolerance"),
        keepbits = args["keepbits"],
        data_types=["input", "target"],
        variables = []
    )

    # every variable key in the scaler (input + target)
    scalers = load_scaler_dict(os.path.expandvars(args["scaler_path"]))
    var_keys = dict.fromkeys(k for dt in block.data_types 
                             for src in scalers[dt].values() for k in src)

    prefixes = tuple(var.rstrip("/") for var in args["variables"])
    rows = []
    tot = np.zeros(3)

    # measure the size of each variable in the preblock raw, losslessly compressed, and lossy compressed
    # sum up each variable's size and add to total
    for var_key in var_keys:
        if not var_key.startswith(prefixes):
            continue
        name = var_key.split("/")[-1]
        if not all(name in root for root in roots):
            continue
        
        k = block._resolve(var_key)

        raw = 0 # uncompressed variable size
        orig = 0 # losslessly compressed variable size
        rounded = 0 # lossy and losslessly compressed variable size

        # iterate through each store 
        for root in roots:
            array = root[name]
            codec = array.compressors[0] # the store's lossless codec
            idx = np.linspace(0, array.shape[0] - 1, n_samples).astype(int)

            # compress each sampled time step, with and without bitrounding
            for i in idx:
                c = np.ascontiguousarray(array[i])
                raw += c.nbytes
                orig += len(codec.encode(c))
                rounded += len(codec.encode(BitRound(keepbits=k).encode(c.copy())))

        # scale the sampled totals up to one full year, so the result is GB/yr
        f = array.nbytes/raw 
        got = np.array([raw, orig, rounded]) * f
        tot += got
        rows.append((name, k, *(got/1e9), 100*(1-(rounded/orig)))) # e.g. (T, 5, 10.33, 6.20, 0.56, 91.03)

    rows.sort(key=lambda r: -r[5]) # sort by percent saved, high to low
    return rows, tot

def print_rows(rows, tot):
    """Print the per-variable table and the totals."""
    print(f"{'var':12s}{'bits':>5}{'raw GB':>9}{'orig GB':>9}{'round GB':>10}{'saved':>8}")
    for r in rows:
        print(f"{r[0]:12s}{r[1]:5d}{r[2]:9.2f}{r[3]:9.2f}{r[4]:10.2f}{r[5]:7.2f}%")
    print(f"\n{'TOTAL':12s}{'':5}{tot[0]/1e9:9.2f}{tot[1]/1e9:9.2f}{tot[2]/1e9:10.2f}"
        f"{100*(1-(tot[2]/tot[1])):7.2f}%")
    print(f"saved {(tot[1]-tot[2])/1e9:.1f} GB/yr = {100*(1-tot[2]/tot[1]):.0f}% of current storage")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to CREDIT config")
    ap.add_argument("-n", "--n-samples", type=int,default=24, help="time steps sampled from each year (default: %(default)s)")
    ap.add_argument("-y", "--n-years", type=int, default=5, help="number of years to sample from (default: %(default)s)")
    ap.add_argument("-s", "--seed", type=int,default=0)
    args = ap.parse_args()
    if not os.path.exists(args.config):
        sys.exit(f"no such file: {args.config}")
    rows, tot = estimate_size_saved(args.config,args.n_samples,args.n_years,args.seed)
    print_rows(rows, tot)