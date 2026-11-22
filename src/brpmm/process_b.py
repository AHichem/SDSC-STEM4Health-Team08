"""Process B: decision rule plus an accepted-only monitor that explains and flags novelty.

Layer 1, decision: ``SetpointDeviationRule`` on aeration_rate, relative to each
batch's own early level. Calibration: 5/5 rejects caught, 0/35 false alarms;
accepted batches never deviate more than 0.36 % from their own level.

Layer 2, review: a PCA monitor fitted on accepted batches only. It catches the
known fault too (AUC 1.0 in CV) and would also flag fault types the rule has
never seen. A monitor alarm on a batch the rule accepts is a *review* flag, not
a rejection: in the test set these are batches started outside the calibrated
operating range (e.g. B_073, starting aeration 8.59 vs at most 8.14 in
calibration) whose own aeration stays flat, i.e. new but acceptable variability.

The gross-outlier detector is not used here: it was tuned on process A, and
B's fast pH and acid/base control loops trip it on normal batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from brpmm.config import BATCH_COL, LABEL_ACCEPTED, LABEL_REJECTED
from brpmm.mspc import PCAMonitor
from brpmm.preprocessing import SegmentParams
from brpmm.rules import SetpointDeviationRule
from brpmm.steady_features import batch_features, sensor_of, to_model_space

SEGMENT_B = SegmentParams(trim_start=50, trim_end=50, min_steady=100)
INITIAL_CONDITIONS = (
    "aeration_rate",
    "fermentation_volume",
    "substrate_concentration",
    "biomass_concentration",
)


def monitor_features(ts: pd.DataFrame) -> pd.DataFrame:
    feats = to_model_space(batch_features(ts, SEGMENT_B))
    return feats.drop(columns=["n_excluded_rows"], errors="ignore")


@dataclass
class ProcessBModel:
    rule: SetpointDeviationRule = field(default_factory=SetpointDeviationRule)

    def fit(self, ts: pd.DataFrame, labels: pd.Series) -> ProcessBModel:
        """``labels`` only selects the accepted batches for the monitor's reference."""
        accepted = labels.index[labels == LABEL_ACCEPTED]
        feats = monitor_features(ts).loc[accepted]
        self.columns_ = list(feats.columns[feats.std() > 1e-9])
        self.monitor_ = PCAMonitor().fit(feats[self.columns_])
        start = ts.groupby(BATCH_COL).first().loc[accepted, list(INITIAL_CONDITIONS)]
        self.start_range_ = start.agg(["min", "max"])
        return self

    def predict(self, ts: pd.DataFrame) -> pd.DataFrame:
        out = self.rule.evaluate(ts)
        feats = monitor_features(ts)[self.columns_]
        mon = self.monitor_.score(feats)
        contrib = (
            self.monitor_.spe_contributions(feats) + self.monitor_.t2_contributions(feats).abs()
        )
        by_sensor = contrib.T.groupby(contrib.columns.map(sensor_of)).sum().T
        out["monitor_score"] = mon["score"]
        out["monitor_alarm"] = mon["alarm"]
        out["monitor_top_sensor"] = by_sensor.idxmax(axis=1)

        start = ts.groupby(BATCH_COL).first()[list(INITIAL_CONDITIONS)]
        lo, hi = self.start_range_.loc["min"], self.start_range_.loc["max"]
        outside = (start < lo) | (start > hi)
        out["start_outside_calibration"] = outside.apply(
            lambda r: ", ".join(c for c in r.index if r[c]), axis=1
        )
        out["label"] = out["rejected"].map({True: LABEL_REJECTED, False: LABEL_ACCEPTED})
        out["review"] = out["monitor_alarm"] & ~out["rejected"]
        out["reason"] = [self._reason(r) for _, r in out.iterrows()]
        return out.drop(columns="rejected")

    def _reason(self, r: pd.Series) -> str:
        if r["label"] == LABEL_REJECTED:
            return (
                f"{self.rule.variable} off its own level by up to {r['max_rel_deviation']:.0%} "
                f"for {int(r['longest_run'])} consecutive samples from step "
                f"{int(r['first_step_beyond'])}"
            )
        if r["review"]:
            new = r["start_outside_calibration"]
            where = f"; started outside calibration range on {new}" if new else ""
            return (
                f"accepted; unusual vs accepted batches (mostly {r['monitor_top_sensor']})"
                f"{where}; own {self.rule.variable} steady (max {r['max_rel_deviation']:.2%})"
            )
        dev = r["max_rel_deviation"]
        return f"accepted; within normal behaviour (max {self.rule.variable} deviation {dev:.2%})"
