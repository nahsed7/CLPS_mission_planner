"""Loose column / schema adapters so any reasonable CSV can feed the engine."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Canonical field -> acceptable header aliases (lowercase, stripped)
ALIASES = {
    "site_id": ["site_id", "id", "site", "landing_site", "siteid"],
    "site_name": ["site_name", "name", "label", "title"],
    "lat": ["latitude_deg", "lat", "latitude", "lat_deg", "lat_dd"],
    "lon": ["longitude_deg", "lon", "longitude", "lon_deg", "long", "lng", "lon_dd"],
    "elev_m": [
        "observer_elevation_m",
        "elevation_m",
        "site_elevation_m",
        "dem_elevation_at_site_m",
        "elev_m",
        "elev",
        "altitude_m",
    ],
    "time_utc": [
        "time_utc",
        "timestamp_utc",
        "datetime_utc",
        "utc",
        "time",
        "timestamp",
        "datetime",
        "date",
    ],
    "azimuth_deg": ["azimuth_deg", "az", "azimuth", "az_deg", "azimut"],
    "horizon_elev_deg": [
        "horizon_elevation_deg",
        "horizon_elev_deg",
        "horizon_deg",
        "elev_horizon_deg",
        "terrain_horizon_deg",
    ],
    "sun_alt": ["sun_alt", "sun_elevation_deg", "sun_elev_deg", "sun_altitude_deg", "sun_el"],
    "sun_az": ["sun_az", "sun_azimuth_deg", "sun_az_deg"],
    "earth_alt": [
        "earth_alt",
        "earth_elevation_deg",
        "earth_center_elevation_deg",
        "earth_elev_deg",
        "earth_el",
    ],
    "earth_az": ["earth_az", "earth_azimuth_deg", "earth_az_deg"],
    "earth_radius": [
        "earth_radius",
        "earth_angular_radius_deg",
        "earth_disc_radius_deg",
    ],
    "earth_visible": ["earth_visible", "dte_visible", "earth_vis", "visible"],
    "sunlit": ["sunlit", "sun_visible", "sun_above_horizon", "illuminated"],
    "height_m": ["height_m", "elevation_m", "z_m", "dem_m", "relief_m"],
    "slope_deg": ["slope_deg", "slope", "slope_degrees"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def resolve_column(columns: Iterable[str], canonical: str, required: bool = True) -> Optional[str]:
    """Map a canonical field name to the actual column present in `columns`."""
    lookup = {_norm(c): c for c in columns}
    for alias in ALIASES.get(canonical, [canonical]):
        key = _norm(alias)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(
            f"Need a column like {ALIASES.get(canonical, [canonical])}; "
            f"found {list(columns)}"
        )
    return None


def rename_to_canonical(df: pd.DataFrame, fields: list[str], required: Optional[list[str]] = None) -> pd.DataFrame:
    """Return a copy with selected fields renamed to canonical names."""
    required = required or []
    mapping = {}
    for field in fields:
        col = resolve_column(df.columns, field, required=field in required)
        if col is not None:
            mapping[col] = field
    out = df.rename(columns=mapping)
    keep = [f for f in fields if f in out.columns]
    extras = [c for c in out.columns if c not in keep]
    return out[keep + extras]


def parse_utc_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s*UTC$", "", regex=True).str.rstrip("Z").str.replace("T", " ")
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)


def read_table(path_or_buf, **kwargs) -> pd.DataFrame:
    """Read CSV/TSV/Parquet from a path or file-like object."""
    if hasattr(path_or_buf, "read"):
        return pd.read_csv(path_or_buf, **kwargs)
    path = Path(path_or_buf)
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    sep = "\t" if name.endswith((".tsv", ".tsv.gz")) else ","
    return pd.read_csv(path, sep=sep, **kwargs)


def infer_step_minutes(times: pd.Series) -> float:
    t = pd.to_datetime(times)
    if len(t) < 2:
        return 60.0
    deltas = t.diff().dropna().dt.total_seconds() / 60.0
    return float(np.median(deltas)) if len(deltas) else 60.0
