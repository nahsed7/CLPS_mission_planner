"""Operational windows: runs, time-to-set, summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def longest_run_hours(flags: np.ndarray, step_min: float) -> float:
    best = cur = 0
    for f in np.asarray(flags, dtype=bool):
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best * step_min / 60.0


def runs_table(
    times,
    flags,
    step_min: float,
    labels=("open", "closed"),
) -> pd.DataFrame:
    flags = np.asarray(flags, dtype=bool)
    if len(flags) == 0:
        return pd.DataFrame(columns=["state", "start_utc", "end_utc", "hours"])
    change = np.flatnonzero(flags[1:] != flags[:-1]) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(flags)]
    times = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    step = pd.Timedelta(minutes=step_min)
    return pd.DataFrame(
        {
            "state": [labels[0] if flags[i] else labels[1] for i in starts],
            "start_utc": [times[i] for i in starts],
            "end_utc": [times[j - 1] + step for j in ends],
            "hours": (ends - starts) * step_min / 60.0,
        }
    )


def time_remaining_hours(flags: np.ndarray, index: int, step_min: float) -> float:
    """Hours until the current True/False state flips (from `index`)."""
    flags = np.asarray(flags, dtype=bool)
    if index < 0 or index >= len(flags):
        return float("nan")
    state = flags[index]
    j = index
    while j < len(flags) and flags[j] == state:
        j += 1
    return (j - index) * step_min / 60.0


def summarise_flags(df: pd.DataFrame, sun_col: str, earth_col: str, step_min: float) -> dict:
    sun = df[sun_col].to_numpy(bool)
    earth = df[earth_col].to_numpy(bool)
    both = sun & earth
    return {
        "sunlit_pct": 100.0 * sun.mean(),
        "earth_visible_pct": 100.0 * earth.mean(),
        "sun_and_earth_pct": 100.0 * both.mean(),
        "longest_sunlight_h": longest_run_hours(sun, step_min),
        "longest_darkness_h": longest_run_hours(~sun, step_min),
        "longest_dte_h": longest_run_hours(earth, step_min),
        "longest_dte_blackout_h": longest_run_hours(~earth, step_min),
        "samples": int(len(df)),
        "step_min": float(step_min),
    }
