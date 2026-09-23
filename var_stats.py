#!/usr/bin/env python3
"""Print summary statistics (mean, median, std, min, max) for selected CESM
variables, sampled from selected years of the zarr stores.

    var_stats.py
    var_stats.py --years 1980 2000 --stride 24
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr

DEFAULT_PATH = (
    "/glade/derecho/scratch/wchapman/b_credit_runs_f32_02/"
    "b.e21.CREDIT_climate_branch_1980_%Y_zmdata_ERA5scaled_zmdata_Qtot.zarr"
)

YEARS = [1980, 1991, 2002, 2012] # sample years spread throughout the record
STRIDE = 48 # keep every Nth 6-hourly step (48 = one every 12 days)

# (variable, "2d"/"3d"), where "3d" means stats are pooled over all levels
VARS = [
    ("U", "3d"),
    ("T", "3d"),
    ("TREFHT", "2d"),
    ("TS", "2d"),
    ("PRECT", "2d"),
    ("TAUX", "2d"),
    ("TAUY", "2d"),
    ("U10", "2d"),
    ("FSUS", "2d"),
    ("FLUS", "2d"),
    ("FSUTOA", "2d"),
    ("FLUT", "2d"),
]

# used when the zarr variable has no units/long_name
FALLBACK = {
    "PRECT": ("m/6h", "Total precipitation rate"),
}


def expand(path, years):
    """
    Return one store path per year. Accepts %Y or {year}, a path with neither
    is returned as is.
    """

    template = path.replace("%Y", "{year}")
    if "{year}" not in template:
        return [path]
    return [template.format(year=year) for year in years]


def describe(ds, name):
    """
    Return (units, long_name) from the variable's attributes, else FALLBACK, 
    else "?"
    """

    units, long_name = FALLBACK.get(name, ("?", "?"))
    attrs = ds[name].attrs
    return attrs.get("units", units), attrs.get("long_name", long_name)


def gather(stores, name, stride):
    """
    Return one flat array holding every 'stride-th' time step of 'name' from
    all stores.
    """

    chunks = []
    for store in stores:
        ds = xr.open_zarr(store)
        # read only sampled steps from disk
        chunks.append(ds[name].isel(time=slice(None, None, stride)).values.ravel())
        ds.close()
    return np.concatenate(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH, 
                    help="zarr path; %%Y or {year} expands over --years" 
                          " (default: CESM2 b_credit_runs_f32_02 stores)")
    ap.add_argument("--years", type=int, nargs="+", default=YEARS, 
                    help="years to sample from stores (default: %(default)s)")
    ap.add_argument("--stride", type=int, default=STRIDE, 
                    help="keep every Nth 6-hourly step (default: %(default)s)")
    args = ap.parse_args()

    stores = expand(args.path, args.years)
    missing = [s for s in stores if not os.path.exists(s)]
    if missing:
        sys.exit("no such store(s):\n  " + "\n  ".join(missing))

    # Check variables and read metadata from the first store only
    # Assumes every year has the same variables
    ds0 = xr.open_zarr(stores[0])
    present = [(name, kind) for name, kind in VARS if name in ds0]
    skipped = [name for name, _ in VARS if name not in ds0]
    meta = {name: describe(ds0, name) for name, _ in present}
    ds0.close()

    if not present:
        sys.exit(f"none of the expected variables are in {stores[0]}")

    records = []
    for name, kind in present:
        a = gather(stores, name, args.stride)
        a = a[np.isfinite(a)] # drop NaN/inf
        # (label, long_name, units, mean, median, std, min, max)
        records.append(
            (
                f"{name} ({kind})",
                meta[name][1],
                meta[name][0],
                float(np.mean(a)),
                float(np.median(a)),
                float(np.std(a)),
                float(np.min(a)),
                float(np.max(a)),
            )
        )
    records.sort(key=lambda r: r[4], reverse=True) # sorts by median, largest to smallest

    # set long_name and unit column widths using longest strings
    width = max(len(long_name) for _, long_name in meta.values())
    uwidth = max(len(units) for units, _ in meta.values())

    label = args.years if len(stores) > 1 else os.path.basename(stores[0])
    print(f"\n{label}  every {args.stride}th 6-hourly step, sorted by median\n")
    header = (
        f"{'Variable':<13}{'long name':<{width + 2}}{'units':<{uwidth + 2}}"
        f"{'mean':>12}{'median':>12}{'std':>12}{'min':>12}{'max':>12}"
    )
    print(header)
    print("-" * len(header))

    for label_, long_name, units, mean, median, std, lo, hi in records:
        print(
            f"{label_:<13}{long_name:<{width + 2}}{units:<{uwidth + 2}}"
            f"{mean:12.4g}{median:12.4g}{std:12.4g}{lo:12.4g}{hi:12.4g}"
        )
    print()

    if skipped:
        print(f"not in this dataset: {', '.join(skipped)}\n")


if __name__ == "__main__":
    main()
