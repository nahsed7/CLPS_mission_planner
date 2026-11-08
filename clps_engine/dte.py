"""Model 2 (generic): Direct-to-Earth geometric visibility for ANY site/dataset."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .horizon import HorizonMask, horizon_at
from .windows import runs_table, time_remaining_hours

# How much of Earth's disc must clear the horizon.
# +1 = any part (NASA / Model 2 default), 0 = centre, -1 = whole disc.
EARTH_RULES = {
    "any_part": 1.0,
    "centre": 0.0,
    "whole_disc": -1.0,
}


def apply_dte_visibility(
    geometry: pd.DataFrame,
    mask: Optional[HorizonMask] = None,
    rule: str = "any_part",
) -> pd.DataFrame:
    df = geometry.copy()
    k = EARTH_RULES.get(rule, 1.0)
    hz = horizon_at(mask, df["earth_az"].to_numpy(float))
    radius = df["earth_radius"].to_numpy(float) if "earth_radius" in df.columns else np.full(len(df), 0.95)
    df["earth_horizon_deg"] = hz
    df["earth_radius"] = radius
    margin = df["earth_alt"].to_numpy(float) + k * radius - hz
    df["visibility_margin_deg"] = margin
    df["earth_visible"] = margin > 0.0
    return df


def dte_report(df: pd.DataFrame, step_min: float) -> dict:
    vis = df["earth_visible"].to_numpy(bool)
    runs = runs_table(df["time_utc"], vis, step_min, ("visible", "blackout"))
    visible = runs[runs["state"] == "visible"]
    black = runs[runs["state"] == "blackout"]
    out = {
        "earth_visible_pct": float(100.0 * vis.mean()),
        "longest_dte_h": float(visible["hours"].max()) if len(visible) else 0.0,
        "longest_dte_blackout_h": float(black["hours"].max()) if len(black) else 0.0,
        "n_blackouts": int(len(black)),
        "n_windows": int(len(visible)),
        "mean_margin_when_visible_deg": float(df.loc[df["earth_visible"], "visibility_margin_deg"].mean())
        if vis.any()
        else float("nan"),
    }
    if len(df):
        out["hours_to_earthset_at_t0"] = time_remaining_hours(vis, 0, step_min) if vis[0] else 0.0
        out["hours_to_earthrise_at_t0"] = time_remaining_hours(~vis, 0, step_min) if not vis[0] else 0.0
    return out


def dte_windows(df: pd.DataFrame, step_min: float) -> pd.DataFrame:
    return runs_table(df["time_utc"], df["earth_visible"], step_min, labels=("visible", "blackout"))
