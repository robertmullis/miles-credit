import os

import numpy as np
import torch

from credit.preblock.base import BasePreblock
from credit.preblock._utils import _parse_variable_selection

_MANTISSA_BITS = 23 # float32 mantissa width
_DERIVE_SEED = 0 # fixed seed so derived keepbits are reproducible
_DERIVE_SAMPLES = 20000 # synthetic samples per level when deriving keepbits from tolerance

class BitRoundTransform(BasePreblock):
    """Bitrounds float32 variables to fewer mantissa bits, the lossy step of bitround compression.
    
    Each value keeps ``keepbits`` of its 23 mantissa bits, using round-to-nearest-even on the raw
    bits (the same scheme as ``numcodecs.BitRound``).

    If ``tolerance`` is set, keepbits is chosen per variable from the fitted scaler:
    the smallest k whose rounding error std is ``<= tolerance * std`` at every level.
    The error is measured on synthetic N(mean, std) samples. Variables missing from
    the scaler, and all variables when ``tolerance`` is null, use ``keepbits``.

    Operates on ``batch[data_type][source][var_key]`` for each requested ``data_type``
    (default: ``["input", "target"]``). Tensors that are not float32 are passed through unchanged.
    
    Config example::

        type: "bitround_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            tolerance: 0.05     # fraction of each variable's std; null = fixed keepbits
            keepbits: 4         # used when tolerance is null or a variable has no scaler entry
            variables:
                - "CESM/prognostic"
                - "CESM/dynamic_forcing"    
    """

    def _load_stats(self):
        """Return {var_key: fitted scaler} from scaler_path, or {} if tolerance is None."""

        from bridgescaler import load_scaler_dict
        if self.tolerance is None:
            return {}

        scalers = load_scaler_dict(self.scaler_path)
        stats = {}
        for data_type in self.data_types:
            for source in scalers[data_type].values():
                for var_key, scaler in source.items():
                    stats.setdefault(var_key, scaler) # first data_type wins
        return stats

    def __init__(self, scaler_path: str, tolerance: float, keepbits: int, variables: list[str], 
                data_types: list[str] = None):
        super().__init__()     

        # validate keepbits even when tolerance is set, because it is the fallback
        if not 0 <= keepbits <= _MANTISSA_BITS: 
            raise ValueError(f"keepbits must be between 0-{_MANTISSA_BITS}")
        
        self.keepbits = keepbits
        self.tolerance = tolerance
        self.variables = variables
        self.variables_expanded = False # partial paths are expanded on first forward()
        self.resolved ={} # var_key -> keepbits, filled on first forward()
        self.data_types = data_types or ["input", "target"]
        self.scaler_path = os.path.expandvars(scaler_path)
        self._stats = self._load_stats()

        invalid = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid:
            raise ValueError(
                f"Invalid data_types {invalid}. "
                f"Valid options are {self.VALID_DATA_TYPES}. "
                f"Preblocks never operate on 'metadata'."
            )

    def _derive_keepbits(self, scaler):
        """Return the smallest keepbits whose rounding error std is <= tolerance * std on every level.

        Each level is tested on synthetic N(mean, std) samples from the scaler, and the
        maximum over levels is returned, so the most demanding level sets the bits.
        """

        # scaler stats, one entry per level (a single entry for 2d variables)
        mu = scaler.mean_x_.double().numpy()
        sd = scaler.var_x_.double().sqrt().numpy()

        rng = np.random.default_rng(_DERIVE_SEED)
        per_level = []
        stats = zip(mu, sd)
        for m, s in stats:
            x = torch.from_numpy(rng.normal(m, s, _DERIVE_SAMPLES).astype(np.float32))
            # try k = 0, 1, 2, ...; the first k that meets the tolerance is the fewest bits needed
            for k in range(_MANTISSA_BITS + 1):
                if float((self._round(x, k) - x).std()) <= self.tolerance * s:
                    per_level.append(k)
                    break
        return max(per_level)

    def _resolve(self, var_key):
        """Return keepbits for one variable: derived if tolerance is set and the scaler has it, else keepbits."""
        
        # `if self.tolerance` is false for 0 as well as None, so tolerance: 0 falls back to fixed keepbits
        if self.tolerance and var_key in self._stats:
            return self._derive_keepbits(self._stats[var_key])
        return self.keepbits

    def _round(self, x, keepbits):
        """Round x to `keepbits` mantissa bits (round-to-nearest-even on the int32 bits)."""
        
        # nothing to do at full precision or for non-float32 tensors
        if keepbits == _MANTISSA_BITS or x.dtype != torch.float32:
            return x
        shift = _MANTISSA_BITS - keepbits # number of low mantissa bits to remove
        i = x.view(torch.int32) # reinterpret the float's bits as an integer
        half = (1 << (shift - 1)) - 1 # just under half of one kept-bit step
        # add half, plus 1 if the last kept bit is odd, so exact ties round to even
        i = i + half + ((i>>shift & 1))

        # clear the dropped bits and reinterpret as float32
        return (i >> shift << shift).view(torch.float32)

    def forward(self, batch: dict):
        """Return a shallow copy of batch with each selected variable bitrounded."""

        if not self.variables_expanded: # first call only: expand partial paths and cache each variable's keepbits
            self.variables = _parse_variable_selection(self.variables, batch, self.data_types)
            for var in self.variables:
                self.resolved[var] = self._resolve(var)
            self.variables_expanded = True
        batch = self._copy_batch(batch)  # shallow copy — avoids mutating the caller's dict
        for var_key in self.variables:
            source = var_key.split("/")[0]  # e.g. "era5" from "era5/prognostic/3d/Q"

            for data_type in self.data_types:
                if data_type not in batch:
                    continue  # data type absent in this batch (e.g. no "target" during inference)
                if source not in batch[data_type]:
                    raise KeyError(f"BitRoundTransform: source '{source}' not found in batch['{data_type}'].")
                if var_key not in batch[data_type][source]:
                    continue  # variable absent in this data type (e.g. statics only exist in "input")

                batch[data_type][source][var_key] = self._round(batch[data_type][source][var_key], self.resolved[var_key])

        return batch

