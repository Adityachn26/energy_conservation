"""Phase 7 — Chronological train / validation / test split.

Why not a random split? With random splitting, the model would train on (say) 15 March and be
tested on 14 March. Lag features of neighbouring hours overlap heavily, so the model effectively
"sees" the test period and the scores look far better than real forecasting performance.
In production we always predict the FUTURE from the PAST — the split must mimic that.

The cutoffs are global calendar timestamps shared by all meters, so every meter's test set is
the same genuinely unseen future period.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from src.config import TRAIN_FRACTION, VALIDATION_FRACTION

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitCutoffs:
    validation_start: pd.Timestamp
    test_start: pd.Timestamp

    def as_dict(self) -> dict:
        return {"validation_start": str(self.validation_start), "test_start": str(self.test_start)}


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitCutoffs]:
    """Split rows by timestamp so that train < validation < test in time.

    Cutoffs are chosen so that roughly ``train_fraction`` of ROWS fall before the validation
    start, etc. All rows sharing a timestamp land in the same split (no meter straddles a cutoff).
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Fractions must satisfy 0 < train < 1 and 0 < validation < 1 - train")

    sorted_ts = df["timestamp"].sort_values().reset_index(drop=True)
    validation_start = sorted_ts.iloc[int(len(sorted_ts) * train_fraction)]
    test_start = sorted_ts.iloc[int(len(sorted_ts) * (train_fraction + validation_fraction))]
    cutoffs = SplitCutoffs(validation_start=validation_start, test_start=test_start)

    train = df.loc[df["timestamp"] < validation_start]
    validation = df.loc[(df["timestamp"] >= validation_start) & (df["timestamp"] < test_start)]
    test = df.loc[df["timestamp"] >= test_start]

    logger.info(
        "Chronological split: train=%s (< %s) | validation=%s | test=%s (>= %s)",
        f"{len(train):,}",
        validation_start,
        f"{len(validation):,}",
        f"{len(test):,}",
        test_start,
    )
    return train, validation, test, cutoffs
