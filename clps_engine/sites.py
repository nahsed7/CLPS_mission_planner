"""Site registry: curated CLPS / south-pole targets + any user CSV / custom coords."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .adapters import read_table, rename_to_canonical


@dataclass
class Site:
    id: str
    name: str
    lat: float
    lon: float
    elev_m: float = 0.0
    source: str = "custom"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# Notable south-pole / CLPS-relevant targets (east-positive lon, degrees).
# Elevations from the team's site_comparison / LOLA-derived products where known.
CURATED = [
    Site("LM1", "Shackleton Rim B", -89.5244, 56.30993, -27.784672, "curated", ["CLPS", "south-pole"]),
    Site("Site06", "Nobile Rim 1", -85.4381, 37.36667, 5946.128418, "curated", ["CLPS", "south-pole"]),
    Site("Site20", "Leibnitz Beta Plateau", -85.4266, 31.74276, 6398.174805, "curated", ["CLPS", "south-pole"]),
    Site("Site23", "Malapert Massif", -85.9948, -0.23578, 4783.629395, "curated", ["CLPS", "south-pole"]),
    Site("Shackleton", "Shackleton crater rim (ref)", -89.9, 0.0, 0.0, "curated", ["reference", "south-pole"]),
    Site("Amundsen", "Amundsen crater floor (ref)", -84.5, 85.0, 0.0, "curated", ["reference", "south-pole"]),
]


def curated_clps_sites() -> list[Site]:
    return list(CURATED)


def site_from_row(row: dict) -> Site:
    sid = str(row["site_id"])
    name = str(row.get("site_name") or sid)
    elev = row.get("elev_m")
    return Site(
        id=sid,
        name=name,
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        elev_m=float(elev) if elev is not None and pd.notna(elev) else 0.0,
        source=str(row.get("source", "csv")),
        tags=list(row.get("tags") or []),
    )


def load_sites_csv(path_or_buf) -> list[Site]:
    """Load any sites table. Requires id/lat/lon (loosely named); name/elev optional."""
    df = read_table(path_or_buf)
    df = rename_to_canonical(df, ["site_id", "site_name", "lat", "lon", "elev_m"], required=["site_id", "lat", "lon"])
    return [site_from_row(r) for r in df.to_dict(orient="records")]


def custom_site(lat: float, lon: float, elev_m: float = 0.0, name: str = "Custom", site_id: str = "CUSTOM") -> Site:
    return Site(id=site_id, name=name, lat=lat, lon=lon, elev_m=elev_m, source="custom", tags=["custom"])


def merge_sites(*groups: Iterable[Site]) -> dict[str, Site]:
    """Later groups override earlier ones with the same id."""
    out: dict[str, Site] = {}
    for group in groups:
        for s in group:
            out[s.id] = s
    return out


def sites_to_dataframe(sites: Iterable[Site]) -> pd.DataFrame:
    return pd.DataFrame([s.to_dict() for s in sites])


def default_sample_sites_path(root: Optional[Path] = None) -> Path:
    root = root or Path(__file__).resolve().parent.parent
    return root / "data" / "sample_sites.csv"
