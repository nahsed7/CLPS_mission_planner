"""
Geometric RF / DTE quality assessment.

Not a full link budget (no EIRP, G/T, or rain). Scores how favourable the
geometry is for a direct-to-Earth pass using elevation margin and Earth disc.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .windows import runs_table


@dataclass
class LinkConfig:
    """Simple geometric thresholds a planner can tune."""

    min_margin_deg: float = 0.0          # need visibility_margin above this
    preferred_margin_deg: float = 2.0    # "good" geometric pass
    min_elevation_deg: float = -1.0      # centre elevation floor (disc rule can be lower)
    antenna_beamwidth_deg: float = 20.0  # for a crude pointing-ease score


def apply_rf_quality(df: pd.DataFrame, cfg: LinkConfig = LinkConfig()) -> pd.DataFrame:
    out = df.copy()
    if "visibility_margin_deg" not in out.columns:
        if "earth_visible" in out.columns:
            out["visibility_margin_deg"] = np.where(out["earth_visible"], 1.0, -1.0)
        else:
            raise ValueError("Need visibility_margin_deg or earth_visible")

    margin = out["visibility_margin_deg"].to_numpy(float)
    # Quality 0..1 while geometrically visible
    q = np.clip(margin / max(cfg.preferred_margin_deg, 1e-6), 0.0, 1.0)
    visible = margin > cfg.min_margin_deg
    # Pointing ease: higher Earth elevation → easier (within beamwidth)
    if "earth_alt" in out.columns:
        elev = out["earth_alt"].to_numpy(float)
        point = np.clip((elev - cfg.min_elevation_deg) / max(cfg.antenna_beamwidth_deg, 1e-6), 0.0, 1.0)
    else:
        point = q
    out["rf_geometry_ok"] = visible
    out["rf_quality"] = np.where(visible, 0.6 * q + 0.4 * point, 0.0)
    out["rf_grade"] = pd.cut(
        out["rf_quality"],
        bins=[-0.01, 0.01, 0.33, 0.66, 1.01],
        labels=["none", "marginal", "usable", "good"],
    )
    return out


def rf_report(df: pd.DataFrame, step_min: float, cfg: LinkConfig = LinkConfig()) -> dict:
    ok = df["rf_geometry_ok"].to_numpy(bool)
    q = df["rf_quality"].to_numpy(float)
    runs = runs_table(df["time_utc"], ok, step_min, ("link", "outage"))
    link = runs[runs["state"] == "link"]
    outage = runs[runs["state"] == "outage"]
    return {
        "config": asdict(cfg),
        "geometry_ok_pct": float(100.0 * ok.mean()),
        "mean_quality_when_ok": float(q[ok].mean()) if ok.any() else 0.0,
        "good_pct": float(100.0 * (df["rf_grade"].astype(str) == "good").mean()),
        "usable_or_better_pct": float(
            100.0 * df["rf_grade"].astype(str).isin(("usable", "good")).mean()
        ),
        "longest_link_h": float(link["hours"].max()) if len(link) else 0.0,
        "longest_outage_h": float(outage["hours"].max()) if len(outage) else 0.0,
        "note": "Geometric only — not a link budget. Real DTE also needs EIRP, G/T, bit rate, and ground-station schedule.",
    }
