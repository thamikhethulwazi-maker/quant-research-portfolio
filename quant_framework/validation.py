"""
quant_framework.validation
===========================
Walk-forward validation with strict, enforced separation between the data used
to *choose* parameters (train / in-sample) and the data used to *measure*
performance (test / out-of-sample).

Design principles
-----------------
1. The optimiser only ever sees training data. The objective is evaluated on
   the training slice; the winning parameters are then applied — untouched —
   to the immediately following test slice.
2. Test slices are non-overlapping and concatenated to form a single, honest
   OOS track record. Nothing in the OOS series was ever used for selection.
3. Two schemes are provided:
     * expanding window  – train set grows through time (more data, but the
       early regime keeps influencing later choices).
     * rolling window    – train set is a fixed-length trailing window (adapts
       to regime change, discards stale data).

The optimiser is deliberately generic: you pass a `param_grid`, a
`backtest_fn(train_data, **params) -> returns` and an `objective_fn(returns) ->
float` to be maximised. This keeps the framework strategy-agnostic.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio


def _param_combinations(param_grid: dict[str, Sequence]) -> list[dict]:
    """Cartesian product of a {name: [values]} grid into a list of dicts."""
    keys = list(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    return [dict(zip(keys, c)) for c in combos]


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series               # concatenated out-of-sample returns
    fold_records: list[dict] = field(default_factory=list)
    scheme: str = "expanding"

    @property
    def chosen_params(self) -> pd.DataFrame:
        return pd.DataFrame([r["best_params"] | {"test_start": r["test_start"],
                                                 "test_end": r["test_end"],
                                                 "oos_obj": r["oos_obj"]}
                             for r in self.fold_records])


class WalkForwardValidator:
    """
    Parameters
    ----------
    n_folds        : number of test folds to carve out of the sample.
    scheme         : 'expanding' or 'rolling'.
    train_span     : (rolling only) length of the trailing train window, in
                     index steps. Ignored for expanding.
    min_train      : minimum number of observations before the first test fold.
    objective_fn   : maps a return series to a scalar to be *maximised*.
                     Defaults to in-sample Sharpe.
    """

    def __init__(self,
                 n_folds: int = 5,
                 scheme: str = "expanding",
                 train_span: int | None = None,
                 min_train: int = 252,
                 objective_fn: Callable[[pd.Series], float] = sharpe_ratio):
        assert scheme in ("expanding", "rolling")
        self.n_folds = n_folds
        self.scheme = scheme
        self.train_span = train_span
        self.min_train = min_train
        self.objective_fn = objective_fn

    def _fold_boundaries(self, n: int) -> list[tuple[int, int, int, int]]:
        """
        Yield (train_start, train_end, test_start, test_end) integer indices.
        test_end is exclusive. Test folds tile the post-min_train region.
        """
        usable = n - self.min_train
        if usable <= self.n_folds:
            raise ValueError("Not enough data for the requested number of folds.")
        fold_size = usable // self.n_folds
        bounds = []
        for i in range(self.n_folds):
            test_start = self.min_train + i * fold_size
            test_end = test_start + fold_size if i < self.n_folds - 1 else n
            if self.scheme == "expanding":
                train_start = 0
            else:  # rolling
                span = self.train_span or self.min_train
                train_start = max(0, test_start - span)
            train_end = test_start  # exclusive -> no overlap with test
            bounds.append((train_start, train_end, test_start, test_end))
        return bounds

    def run(self,
            data,
            param_grid: dict[str, Sequence],
            backtest_fn: Callable[..., pd.Series],
            slicer: Callable[[object, int, int], object]) -> WalkForwardResult:
        """
        Execute the walk-forward.

        Parameters
        ----------
        data        : arbitrary data container (DataFrame, tuple of Series, ...)
        param_grid  : {param_name: [values]} search space.
        backtest_fn : (data_slice, **params) -> pd.Series of returns.
        slicer      : (data, start_idx, end_idx) -> data_slice. Lets the caller
                      control how integer bounds map onto their data structure.
        """
        # Establish the length of the sample from the slicer on a probe.
        n = _infer_length(data)
        bounds = self._fold_boundaries(n)
        combos = _param_combinations(param_grid)

        oos_pieces: list[pd.Series] = []
        records: list[dict] = []

        for (tr_s, tr_e, te_s, te_e) in bounds:
            train_slice = slicer(data, tr_s, tr_e)

            best_obj, best_params = -np.inf, combos[0]
            for params in combos:
                is_ret = backtest_fn(train_slice, **params)
                obj = self.objective_fn(is_ret)
                if np.isfinite(obj) and obj > best_obj:
                    best_obj, best_params = obj, params

            # Apply winning params to the untouched OOS slice.
            test_slice = slicer(data, te_s, te_e)
            oos_ret = backtest_fn(test_slice, **best_params)
            oos_pieces.append(oos_ret)

            records.append({
                "best_params": best_params,
                "is_obj": float(best_obj),
                "oos_obj": float(self.objective_fn(oos_ret)),
                "test_start": _index_at(data, te_s),
                "test_end": _index_at(data, te_e - 1),
            })

        oos_returns = pd.concat(oos_pieces).sort_index()
        oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
        return WalkForwardResult(oos_returns=oos_returns,
                                 fold_records=records,
                                 scheme=self.scheme)


# ---------------------------------------------------------------------------
# Length / index helpers for common data containers
# ---------------------------------------------------------------------------
def _infer_length(data) -> int:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return len(data)
    if isinstance(data, (tuple, list)):
        return len(data[0])
    raise TypeError(f"Cannot infer length of {type(data)}; pass a custom probe.")


def _index_at(data, i: int):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.index[i]
    if isinstance(data, (tuple, list)):
        return data[0].index[i]
    return i


# ---------------------------------------------------------------------------
# Convenient default slicers
# ---------------------------------------------------------------------------
def dataframe_slicer(df: pd.DataFrame, s: int, e: int) -> pd.DataFrame:
    return df.iloc[s:e]


def tuple_of_series_slicer(t: Sequence[pd.Series], s: int, e: int) -> tuple:
    return tuple(x.iloc[s:e] for x in t)
