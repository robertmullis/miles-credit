"""Create a grid of per-variable NRMSE plots across different forecasts, by comparing
credit rollout output to source data.

    python plot_rollout_compare.py
    python plot_rollout_compare.py -rf
    python plot_rollout_compare.py -f path/to/forecasts -v TS PRECT FSUS

"""

import glob
import os
import argparse
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# change paths to match layout and available runs
FORECASTS = os.path.expandvars("/glade/derecho/scratch/$USER/miles-credit/forecasts")
SOURCE = ("/glade/derecho/scratch/wchapman/b_credit_runs_f32_02/"
          "b.e21.CREDIT_climate_branch_1980_{year}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
CACHE = os.path.expandvars("/glade/derecho/scratch/$USER/miles-credit/forecasts/scores")
OUT = os.path.expandvars("/glade/work/$USER/miles-credit/camulator-plots/rollout-plots")

RUNS = {"gen2": "my_camulator_gen2_rollout",
        "tol_02": "my_camulator_lossy_tol_02_rollout",
        "tol_05": "my_camulator_lossy_tol_05_rollout"}
COLORS = {"gen2": "#2a78d6", "tol_02": "#e07a2b", "tol_05": "#2e9e63"}

def score_dir(init_dir, source, variables=None):
    """Return per-variable forecast RMSE and source STD for one initialization."""

    # forecast output
    fc = xr.open_mfdataset(f"{init_dir}/*/*.nc", combine="by_coords")
    years = sorted({str(t)[:4] for t in fc.time.values})

    # source data
    truth = xr.open_mfdataset([source.format(year=y) for y in years], engine="zarr",
                              combine="by_coords").sel(time=fc.time.values)

    # area weights and the dims to reduce over
    w = np.cos(np.deg2rad(fc.latitude))
    dims = ("latitude", "longitude")

    # calculate RMSE and truth STD for each variable
    names = variables or [v for v in fc.data_vars if v in truth]
    out = {}
    for v in names:
        diff = fc[v] - truth[v]
        out[f"{v}_rmse"] = np.sqrt((diff ** 2).weighted(w).mean(dim=dims))
        out[f"{v}_truth_std"] = truth[v].weighted(w).std(dim=dims)

    # re-index the time as hours since initialization
    ds = xr.Dataset(out).compute()
    step = float((fc.time[1] - fc.time[0]).values / np.timedelta64(1, "h"))
    lead = (fc.time - fc.time[0]).values / np.timedelta64(1, "h") + step
    return ds.assign_coords(lead_hours=("time", lead)).swap_dims(time="lead_hours").drop_vars("time")

def score_config(tag, forecasts, source, cache, variables=None, refresh=False):
    """Score one configuration's initializations and cache the result as NetCDF."""

    # reuse the cached scores unless refresh is set
    os.makedirs(cache, exist_ok=True)
    cached = f"{cache}/{tag}.nc"
    if os.path.exists(cached) and not refresh:
        return xr.open_dataset(cached)
    
    # score each initialization directory
    parts = []
    for d in sorted(glob.glob(f"{forecasts}/{tag}/*")):
        print(f"===Scoring {d}===")
        parts.append(score_dir(d, source, variables).assign_coords(init=os.path.basename(d)))
    
    # save scores by initialization times
    ds = xr.concat(parts, dim="init")
    ds.to_netcdf(cached)
    return ds

def plot(scores, fname, variables=None, out=OUT):
    """Plot a grid of forecast variables, sorted by maximum NRMSE."""
    
    ref = next(iter(scores.values()))
    names = list(variables) if variables else [v[:-5] for v in ref.data_vars if v.endswith("_rmse")]
    names.sort(
    key=lambda n: -float((ref[f"{n}_rmse"].mean("init") / float(ref[f"{n}_truth_std"].mean())).max())
    )

    ncol = int(len(names)**.5) if (len(names)**.5)%1 == 0 else int(len(names)**.5)+1
    nrow = int(np.ceil(len(names)/ ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), sharex=True, squeeze=False)

    for ax, v in zip(axes.ravel(), names):
        for k, s in scores.items():
            y = s[f"{v}_rmse"]
            y = y / float(s[f"{v}_truth_std"].mean()) # normalize RMSE by dividing by mean STD
            lo, hi = y.quantile([0.0, 1.0], dim="init") # min and max across initializations
            ax.fill_between(y.lead_hours, lo, hi, color=COLORS[k], alpha=0.10, lw=0)
            ax.plot(y.lead_hours, y.mean("init"), color=COLORS[k], lw=1.5, label=k)
            
        ax.axhline(1.0, color="0.5", lw=2) # 1.0 = error equal to the field's spatial variability (no skill)
        ax.set_title(v, loc="left", fontsize=10)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.tick_params(which="major", length=6, width=1.2)
        ax.tick_params(which="minor", length=3, width=0.8)

    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for i, ax in enumerate(axes.ravel()[:len(names)]):
        if i + ncol >= len(names):
            ax.set_xlabel("Forecast Time (h)")
            ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE / STD")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0 - 0.10 / fig.get_figheight()),
                     frameon=True, ncol=len(scores), edgecolor="0.3", framealpha=1.0,
                     borderaxespad=0.0)
    for line in leg.get_lines():
        line.set_linewidth(3)

    fig.tight_layout()
    fig.canvas.draw()
    top = leg.get_window_extent().transformed(fig.transFigure.inverted()).y1
    fig.suptitle("Per-Variable NRMSE Comparison", fontsize=16, va="bottom",
                 y=top + 0.15 / fig.get_figheight())

    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--forecasts" , default=FORECASTS, 
                    help="path to forecast directory (default: %(default)s)")
    ap.add_argument("-s", "--source", default=SOURCE, help="path to source data (default: %(default)s)")
    ap.add_argument("-c", "--cache", default=CACHE, help="path to save scores to (default: %(default)s)")
    ap.add_argument("-v", "--variables", default=None, nargs="+", help="variables to plot (default: all)")
    ap.add_argument("-rf", "--refresh", action="store_true", 
                    help="rescore even if a cache exists (default: False)")
    ap.add_argument("-o", "--out", default=OUT, help="directory for the output plot (default: %(default)s)")
    args = ap.parse_args()

    scores = {k: score_config(v, args.forecasts, args.source, args.cache, refresh=args.refresh) for k, v in RUNS.items()}
    tag = "all" if not args.variables else "_".join(args.variables)
    plot(scores, f"rollout_compare_{tag}.png", args.variables, args.out)