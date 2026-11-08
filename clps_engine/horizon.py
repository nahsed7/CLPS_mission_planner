"""Terrain horizon masks from any CSV (per-site or combined profiles)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .adapters import read_table, rename_to_canonical

HorizonMask = Tuple[np.ndarray, np.ndarray]  # azimuth_deg, horizon_elev_deg


def load_horizon_table(path_or_buf) -> pd.DataFrame:
    df = read_table(path_or_buf)
    df = rename_to_canonical(
        df,
        ["site_id", "azimuth_deg", "horizon_elev_deg"],
        required=["azimuth_deg", "horizon_elev_deg"],
    )
    return df


def mask_for_site(df: pd.DataFrame, site_id: Optional[str] = None) -> Optional[HorizonMask]:
    if df is None or len(df) == 0:
        return None
    sub = df
    if site_id is not None and "site_id" in df.columns:
        sub = df[df["site_id"].astype(str) == str(site_id)]
        if len(sub) == 0:
            return None
    sub = sub.sort_values("azimuth_deg")
    az = sub["azimuth_deg"].to_numpy(float)
    el = sub["horizon_elev_deg"].to_numpy(float)
    if len(az) < 8 or not np.all(np.isfinite(el)):
        raise ValueError("Horizon data must be finite with at least 8 azimuth samples")
    return az, el


def load_horizon_for_site(
    site_id: str,
    search_dirs: Optional[list] = None,
    combined_name: str = "horizon_profiles.csv",
) -> Optional[HorizonMask]:
    """Find a horizon mask on disk: combined profiles first, then <id>_horizon.csv."""
    search_dirs = search_dirs or []
    for d in search_dirs:
        d = Path(d)
        combined = d / combined_name
        if combined.exists():
            df = load_horizon_table(combined)
            m = mask_for_site(df, site_id)
            if m is not None:
                return m
        for cand in (
            d / f"{site_id.lower()}_horizon.csv",
            d / f"{site_id}_horizon.csv",
            d / f"{site_id.lower()}.csv",
        ):
            if cand.exists():
                return mask_for_site(load_horizon_table(cand), site_id)
    return None


def horizon_at(mask: Optional[HorizonMask], az: np.ndarray) -> np.ndarray:
    """Interpolate horizon elevation (deg) at given azimuths. Flat 0° if no mask."""
    az = np.asarray(az, dtype=float)
    if mask is None:
        return np.zeros_like(az)
    return np.interp(az, mask[0], mask[1], period=360.0)


def flat_mask() -> HorizonMask:
    az = np.arange(0, 360, 1.0)
    return az, np.zeros_like(az)
