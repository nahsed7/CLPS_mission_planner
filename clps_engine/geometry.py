"""Sun/Earth local geometry: live Skyfield OR ingest precomputed CSVs / NPZ."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .adapters import infer_step_minutes, parse_utc_series, read_table, rename_to_canonical, resolve_column
from .sites import Site

EARTH_RADIUS_KM = 6378.137
KERNEL_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "kernels"


def _loader(kernel_dir: Path):
    from skyfield.api import Loader
    from skyfield.planetarylib import PlanetaryConstants

    loader = Loader(str(kernel_dir))
    ts = loader.timescale()
    eph = loader("de421.bsp")
    pc = PlanetaryConstants()
    pc.read_text(loader("moon_080317.tf"))
    pc.read_text(loader("pck00008.tpc"))
    pc.read_binary(loader("moon_pa_de421_1900-2050.bpc"))
    frame = pc.build_frame_named("MOON_ME_DE421")
    return ts, eph, pc, frame


_CACHE = {}


def load_kernels(kernel_dir: Optional[Path] = None):
    kernel_dir = Path(kernel_dir or KERNEL_DIR_DEFAULT)
    key = str(kernel_dir)
    if key not in _CACHE:
        _CACHE[key] = _loader(kernel_dir)
    return _CACHE[key]


def compute_geometry(
    site: Site,
    start_iso: str,
    hours: float,
    step_min: float = 60.0,
    kernel_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute Sun/Earth alt-az at a lunar surface site over a time window."""
    n = max(2, int(round(hours * 60 / step_min)))
    return compute_geometry_n(site, start_iso, n, step_min, kernel_dir)


def compute_geometry_n(
    site: Site,
    start_iso: str,
    n: int,
    step_min: float,
    kernel_dir: Optional[Path] = None,
) -> pd.DataFrame:
    ts, eph, pc, frame = load_kernels(kernel_dir)
    start = datetime.fromisoformat(start_iso.replace("Z", ""))
    t0 = ts.utc(start.year, start.month, start.day, start.hour, start.minute, start.second)
    t = ts.tt_jd(t0.tt + np.arange(n) * step_min / 1440.0)

    place = eph["moon"] + pc.build_latlon_degrees(frame, site.lat, site.lon, elevation_m=site.elev_m)
    obs = place.at(t)
    sun_alt, sun_az, _ = obs.observe(eph["sun"]).apparent().altaz()
    earth_alt, earth_az, earth_dist = obs.observe(eph["earth"]).apparent().altaz()
    radius = np.degrees(np.arcsin(np.minimum(1.0, EARTH_RADIUS_KM / earth_dist.km)))

    return pd.DataFrame(
        {
            "time_utc": pd.to_datetime(t.utc_datetime()).tz_localize(None),
            "sun_alt": sun_alt.degrees,
            "sun_az": sun_az.degrees,
            "earth_alt": earth_alt.degrees,
            "earth_az": earth_az.degrees,
            "earth_radius": radius,
            "site_id": site.id,
        }
    )


def ingest_geometry_csv(path_or_buf, site_id: Optional[str] = None) -> pd.DataFrame:
    """
    Accept any precomputed geometry table (SPICE/datakit/Horizons exports).

    Required loosely: time + sun_alt + sun_az + earth_alt + earth_az.
    Optional: earth_radius, site_id, sunlit, earth_visible.
    """
    df = read_table(path_or_buf)
    fields = [
        "time_utc",
        "sun_alt",
        "sun_az",
        "earth_alt",
        "earth_az",
        "earth_radius",
        "site_id",
        "sunlit",
        "earth_visible",
    ]
    df = rename_to_canonical(
        df,
        fields,
        required=["time_utc", "sun_alt", "sun_az", "earth_alt", "earth_az"],
    )
    df["time_utc"] = parse_utc_series(df["time_utc"])
    if site_id is not None:
        if "site_id" in df.columns:
            df = df[df["site_id"].astype(str) == str(site_id)].copy()
        df["site_id"] = site_id
    if "earth_radius" not in df.columns:
        # ~0.95 deg typical from lunar surface; better than failing.
        df["earth_radius"] = 0.95
    return df.reset_index(drop=True)


def ingest_geometry_npz(path: Path, site: Site) -> pd.DataFrame:
    """
    Ingest datakit spice ephemeris_*.npz (positions from Moon centre, km)
    and convert to local alt-az at the given site.
    """
    data = np.load(path)
    utc = data["utc"]
    if isinstance(utc[0], bytes):
        utc = [u.decode() for u in utc]

    moon_r_km = 1737.4
    lat, lon = np.radians(site.lat), np.radians(site.lon)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    site_vec = (moon_r_km + site.elev_m / 1000.0) * up

    def local_angles(body_km: np.ndarray):
        topo = body_km - site_vec
        e, n, u = topo @ east, topo @ north, topo @ up
        el = np.degrees(np.arctan2(u, np.hypot(e, n)))
        az = np.degrees(np.arctan2(e, n)) % 360.0
        dist = np.linalg.norm(topo, axis=1)
        return el, az, dist

    sun_el, sun_az, _ = local_angles(data["sun_km"])
    earth_el, earth_az, earth_dist = local_angles(data["earth_km"])
    radius = np.degrees(np.arcsin(np.minimum(1.0, EARTH_RADIUS_KM / earth_dist)))

    return pd.DataFrame(
        {
            "time_utc": parse_utc_series(pd.Series(utc)),
            "sun_alt": sun_el,
            "sun_az": sun_az,
            "earth_alt": earth_el,
            "earth_az": earth_az,
            "earth_radius": radius,
            "site_id": site.id,
        }
    )


def geometry_from_any(
    site: Site,
    *,
    start_iso: Optional[str] = None,
    hours: Optional[float] = None,
    step_min: float = 60.0,
    geometry_csv=None,
    geometry_npz: Optional[Path] = None,
    kernel_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Prefer uploaded/precomputed data; fall back to live ephemeris."""
    if geometry_csv is not None:
        return ingest_geometry_csv(geometry_csv, site_id=site.id)
    if geometry_npz is not None:
        return ingest_geometry_npz(Path(geometry_npz), site)
    if start_iso is None or hours is None:
        raise ValueError("Provide geometry_csv / geometry_npz, or start_iso + hours for live compute.")
    return compute_geometry(site, start_iso, hours, step_min, kernel_dir)


def step_minutes_of(df: pd.DataFrame) -> float:
    return infer_step_minutes(df["time_utc"])
