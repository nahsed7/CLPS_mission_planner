"""Load Eshaal's precomputed Model 1–3 CSVs so the device works with zero kernels."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .adapters import parse_utc_series, read_table
from .sites import Site, curated_clps_sites

DATA = Path(__file__).resolve().parent.parent / "data"


def solar_timeline_path() -> Path:
    return DATA / "solar" / "solar_timeline_all_sites.csv"


def dte_timeline_path(site_id: str) -> Path:
    return DATA / "dte" / f"{site_id}_dte_timeline.csv"


def model3_path() -> Path:
    return DATA / "model3_site_comparison.csv"


def load_solar_timeline(site_id: Optional[str] = None) -> pd.DataFrame:
    path = solar_timeline_path()
    if not path.exists():
        raise FileNotFoundError(path)
    df = read_table(path)
    df["time_utc"] = parse_utc_series(df["time_utc"] if "time_utc" in df.columns else df.iloc[:, 2])
    if "sunlight_visible" in df.columns:
        df["sunlit"] = df["sunlight_visible"].astype(str).str.lower().isin(("true", "1", "yes"))
    if site_id is not None:
        df = df[df["site_id"].astype(str) == str(site_id)].copy()
    return df.reset_index(drop=True)


def load_dte_timeline(site_id: str) -> pd.DataFrame:
    path = dte_timeline_path(site_id)
    if not path.exists():
        raise FileNotFoundError(path)
    df = read_table(path)
    tcol = "timestamp_utc" if "timestamp_utc" in df.columns else "time_utc"
    df["time_utc"] = parse_utc_series(df[tcol])
    df["earth_visible"] = df["earth_visible"].astype(str).str.lower().isin(("true", "1", "yes")) | (
        pd.to_numeric(df["earth_visible"], errors="coerce").fillna(0) > 0
    )
    # Canonical geometry-ish columns for the power/UI pipeline
    rename = {
        "earth_azimuth_deg": "earth_az",
        "earth_center_elevation_deg": "earth_alt",
        "earth_angular_radius_deg": "earth_radius",
        "terrain_horizon_deg": "earth_horizon_deg",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df.reset_index(drop=True)


def load_model3() -> pd.DataFrame:
    path = model3_path()
    if not path.exists():
        raise FileNotFoundError(path)
    return read_table(path).set_index("site_id")


def load_solar_summary() -> pd.DataFrame:
    path = DATA / "solar" / "solar_site_comparison.csv"
    if not path.exists():
        path = DATA / "solar_site_comparison.csv"
    return read_table(path)


def load_dte_summary() -> pd.DataFrame:
    path = DATA / "dte" / "four_site_dte_summary.csv"
    if not path.exists():
        path = DATA / "four_site_dte_summary.csv"
    return read_table(path)


def merge_offline_geometry(site: Site) -> pd.DataFrame:
    """
    Build a timeline usable by power/UI from Model 1 + Model 2 CSVs.

    Model 1 has sunlit flags but not sun alt/az.
    Model 2 has Earth geometry + horizon toward Earth + visibility.
    We synthesize sun_alt proxy from sunlit (for power: lit→assume 5° elev, dark→-5°)
    when true ephemeris isn't available — labelled as offline-approx in metadata.
    """
    solar = load_solar_timeline(site.id)
    dte = load_dte_timeline(site.id)

    # Align on nearest timestamp (both are 10-min grids from same window)
    solar = solar.sort_values("time_utc")
    dte = dte.sort_values("time_utc")
    n = min(len(solar), len(dte))
    s = solar.iloc[:n].reset_index(drop=True)
    d = dte.iloc[:n].reset_index(drop=True)

    sunlit = s["sunlit"].to_numpy(bool)
    # Optimistic power proxy: when sunlit, use a mid-latitude polar elevation estimate
    # from visibility margin isn't available for Sun — use 3° typical south-pole grazing sun.
    sun_alt = np.where(sunlit, 3.0, -5.0)
    sun_az = d["earth_az"].to_numpy(float)  # unknown; unused when dual_axis / lit-gated

    out = pd.DataFrame(
        {
            "time_utc": d["time_utc"],
            "sun_alt": sun_alt,
            "sun_az": sun_az,
            "earth_alt": d["earth_alt"].to_numpy(float),
            "earth_az": d["earth_az"].to_numpy(float),
            "earth_radius": d["earth_radius"].to_numpy(float) if "earth_radius" in d.columns else 0.95,
            "earth_horizon_deg": d["earth_horizon_deg"].to_numpy(float) if "earth_horizon_deg" in d.columns else 0.0,
            "visibility_margin_deg": d["visibility_margin_deg"].to_numpy(float)
            if "visibility_margin_deg" in d.columns
            else 0.0,
            "sunlit": sunlit,
            "earth_visible": d["earth_visible"].to_numpy(bool),
            "site_id": site.id,
            "geometry_source": "offline_model1_model2",
        }
    )
    return out


def offline_team_scorecard() -> pd.DataFrame:
    """One row per site from the team's published CSV summaries + Model 3."""
    sites = {s.id: s for s in curated_clps_sites() if s.id in ("LM1", "Site06", "Site20", "Site23")}
    solar = load_solar_summary()
    dte = load_dte_summary()
    m3 = load_model3()

    # Normalise solar columns
    solar = solar.rename(
        columns={
            "sunlight_percent": "sunlit_pct",
            "longest_sunlight_hours": "longest_sunlight_h",
            "longest_darkness_hours": "longest_darkness_h",
        }
    )
    # DTE summary column names vary — detect loosely
    dte_cols = {c.lower(): c for c in dte.columns}
    rows = []
    for sid, site in sites.items():
        row = {"site_id": sid, "site_name": site.name, "lat": site.lat, "lon": site.lon}
        sr = solar[solar["site_id"].astype(str) == sid]
        if len(sr):
            r = sr.iloc[0]
            row["sunlit_pct"] = float(r.get("sunlit_pct", r.get("sunlight_percent", np.nan)))
            row["longest_sunlight_h"] = float(r.get("longest_sunlight_h", r.get("longest_sunlight_hours", np.nan)))
            row["longest_darkness_h"] = float(r.get("longest_darkness_h", r.get("longest_darkness_hours", np.nan)))
        dr = dte[dte["site_id"].astype(str) == sid] if "site_id" in dte.columns else dte.iloc[0:0]
        if len(dr):
            r = dr.iloc[0]
            if "visibility_percent" in r.index:
                row["earth_visible_pct"] = float(r["visibility_percent"])
            if "longest_visible_run_minutes" in r.index:
                row["longest_dte_h"] = float(r["longest_visible_run_minutes"]) / 60.0
            if "longest_blackout_minutes" in r.index:
                row["longest_dte_blackout_h"] = float(r["longest_blackout_minutes"]) / 60.0
        if sid in m3.index:
            r = m3.loc[sid]
            row["pct_slope_le_10deg"] = float(r["pct_slope_le_10deg"])
            row["flat_circle_le_10deg_m"] = float(r["largest_flat_circle_le_10deg_diameter_m"])
            row["relief_m"] = float(r["relief_m"])
            row["terrain_score"] = float(r["terrain_score_provisional"])
            row["terrain_rank"] = int(r["terrain_rank_provisional"])
        rows.append(row)
    return pd.DataFrame(rows)
