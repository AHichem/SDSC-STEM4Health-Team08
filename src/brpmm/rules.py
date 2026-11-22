"""Deterministic, physically interpretable decision rules.

Process B (fed-batch fermentation): every rejected calibration batch shows the
same fault, aeration falling to 0.70x of the batch's *own* early level for
steps 500-1250 (751 samples), while every accepted batch stays within 0.35% of
its own level. The absolute level differs a lot between good batches (about 4.8
to 8.1), so the rule compares each batch with itself, not with a global range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from brpmm.config import BATCH_COL
from brpmm.data import TIME_COL


@dataclass(frozen=True)
class SetpointDeviationRule:
    variable: str = "aeration_rate"
    baseline_steps: int = 500  # the batch's own reference level: median of its first N steps
    rel_tolerance: float = 0.10  # deviations up to 10% are normal (accepted max: 0.35%)
    min_run: int = 100  # consecutive samples beyond tolerance needed to reject (faults: 751)

    def evaluate(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Per batch: baseline, largest relative deviation, longest run beyond tolerance,
        decision."""
        rows = []
        for b, d in ts.groupby(BATCH_COL, sort=False):
            d = d.sort_values(TIME_COL)
            x = d[self.variable].to_numpy()
            base = float(np.median(x[: self.baseline_steps]))
            dev = np.abs(x / base - 1.0)
            beyond = dev > self.rel_tolerance
            run = longest_run(beyond)
            start = int(d[TIME_COL].to_numpy()[np.argmax(beyond)]) if beyond.any() else None
            rows.append(
                {
                    BATCH_COL: b,
                    "baseline": base,
                    "max_rel_deviation": float(dev.max()),
                    "longest_run": run,
                    "first_step_beyond": start,
                    "rejected": run >= self.min_run,
                }
            )
        return pd.DataFrame(rows).set_index(BATCH_COL)


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest stretch of consecutive True values."""
    if not mask.any():
        return 0
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())
