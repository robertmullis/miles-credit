# CAMulator Bitrounding

Train and evaluate CAMulator with BitRound lossy compression, then measure the storage
saved against the accuracy lost.

```bash
module load conda
conda activate credit-derecho
cd /glade/work/mullis/miles-credit
```

## 1. Create the configs

`create_lossy_config.py` writes a training config and its matching evaluation config
into `camulator_configs/lossy/`.

```bash
python create_lossy_config.py -t <tolerance>  # percent of each variable's std, e.g. -t 5 -> 5%
python create_lossy_config.py -k <keepbits>    # fixed mantissa bits, e.g. -k 6
```

| Flag | Meaning | Default |
|---|---|---|
| `-t`, `--tolerance` | keepbits derived per variable, given as a percent | none |
| `-k`, `--keepbits` | fixed bits, used when no tolerance is given | 4 |
| `-ne`, `--num_epochs` | epochs per job | 10 |
| `-e`, `--epochs` | total epochs | 30 |
| `-tc`, `--template` | config to copy from | `/glade/work/mullis/miles-credit/camulator_configs/lossy/my_camulator_lossy_tol_05.yml` |

`-t 5` produces `my_camulator_lossy_tol_05.yml` and `my_camulator_lossy_tol_05_eval.yml`.
Without `-t`, the tag is `<k>bits` and the config sets `tolerance: null`.

## 2. Submit the training run

```bash
credit submit --cluster derecho -c camulator_configs/lossy/my_camulator_lossy_tol_05.yml \
    --gpus 4 --nodes 4 --walltime 07:00:00 --dry-run
```

Drop `--dry-run` to submit. Jobs are chained with `afterok` based on `epochs / num_epoch`.
To continue a finished run, raise `epochs` and add `--reload --chain <n>`.

## 3. Inspect the training log

```bash
python sort_log.py <run>/training_log.csv              # one epoch's metrics, sorted
python plot_log.py <run>/training_log.csv --split valid  # r2 per epoch, saved as PNG
```

Runs live in `/glade/derecho/scratch/$USER/miles-credit/CREDIT_runs/`.

## 4. Run the evaluation

The eval config has no bitround preblock, a learning rate of 0 and one training batch,
so the job only scores 400 validation batches. Copy the trained weights into its
`save_loc` first.

```bash
R=/glade/derecho/scratch/$USER/miles-credit/CREDIT_runs
mkdir -p $R/camulator_lossy_tol_05_eval
cp $R/camulator_lossy_tol_05/checkpoint.pt $R/camulator_lossy_tol_05_eval/

credit submit --cluster derecho -c camulator_configs/lossy/my_camulator_lossy_tol_05_eval.yml \
    --gpus 1 --walltime 01:00:00
```

Each evaluation needs its own directory. An eval writes a single epoch-0 row, so
rerunning one in place overwrites the previous result.

## 5. Compare against the baseline

```bash
python compare_runs.py \
    -c camulator_configs/lossy/my_camulator_lossy_tol_05.yml \
    -l $R/camulator_lossy_tol_05_eval/training_log.csv \
    --per-var
```

This prints GB/yr, loss, and r2 for the rounded run beside the baseline, plus the percent
change in each. The baseline defaults to the gen2 eval log; pass `-b` for a different one,
and `-bc` with it to compare two rounded configs against each other.

## Storage estimate on its own

```bash
python bitround_estimate_size.py -c camulator_configs/lossy/my_camulator_lossy_tol_05.yml
```

Samples time steps from the CESM zarr stores and reports, per variable, the uncompressed,
losslessly compressed and bitrounded sizes in GB/yr. No trained model needed.

## Notes

- The `bitround_transform` preblock lives in `credit/preblock/bitround.py` and is
  registered in `credit/preblock/__init__.py`. A pull that replaces that file drops the
  registration, and every lossy job then fails at startup with
  `unknown preblock type 'bitround_transform'`.
- `tolerance` is a required argument, so a config that doesn't use it must still set
  `tolerance: null`.
- `keepbits` is a required argument. Lossy runs with a `tolerance` > 0 still use keepbits 
   in the case that tolerance can not be applied to a variable.
- Validation loss measures single-step error only. Use `credit rollout` to see whether
  the rounding error compounds over a forecast.
