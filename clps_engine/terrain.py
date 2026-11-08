"""Model 3: scientifically traceable terrain screening from real DEM products.

Terrain grids preserve nodata. Metrics are computed only from valid cells inside
the requested footprint; no missing pixels are silently replaced with mean terrain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .adapters import read_table, rename_to_canonical


def _masked_gradient(a: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    """Finite differences that do not bridge nodata gaps."""
    z = np.asarray(a, dtype=np.float64)
    out = np.full(z.shape, np.nan, dtype=np.float64)
    if z.shape[axis] < 2:
        return out
    mid = [slice(None)] * z.ndim
    prev = [slice(None)] * z.ndim
    nxt = [slice(None)] * z.ndim
    mid[axis] = slice(1, -1)
    prev[axis] = slice(0, -2)
    nxt[axis] = slice(2, None)
    m, p, n = tuple(mid), tuple(prev), tuple(nxt)
    vp, vn = np.isfinite(z[p]), np.isfinite(z[n])
    both = vp & vn & np.isfinite(z[m])
    temp = np.full(z[m].shape, np.nan, dtype=np.float64)
    temp[both] = (z[n][both] - z[p][both]) / (2.0 * spacing)
    only_p = vp & ~vn & np.isfinite(z[m])
    temp[only_p] = (z[m][only_p] - z[p][only_p]) / spacing
    only_n = vn & ~vp & np.isfinite(z[m])
    temp[only_n] = (z[n][only_n] - z[m][only_n]) / spacing
    out[m] = temp

    first = [slice(None)] * z.ndim
    second = [slice(None)] * z.ndim
    first[axis], second[axis] = 0, 1
    f, s = tuple(first), tuple(second)
    ok = np.isfinite(z[f]) & np.isfinite(z[s])
    out[f][ok] = (z[s][ok] - z[f][ok]) / spacing

    last = [slice(None)] * z.ndim
    before = [slice(None)] * z.ndim
    last[axis], before[axis] = -1, -2
    l, b = tuple(last), tuple(before)
    ok = np.isfinite(z[l]) & np.isfinite(z[b])
    out[l][ok] = (z[l][ok] - z[b][ok]) / spacing
    return out


def derive_slope(height: np.ndarray, mpp_x: float, mpp_y: Optional[float] = None) -> np.ndarray:
    """Derive slope in degrees while retaining nodata and gap boundaries."""
    if mpp_y is None:
        mpp_y = mpp_x
    dzdy = _masked_gradient(height, float(mpp_y), axis=0)
    dzdx = _masked_gradient(height, float(mpp_x), axis=1)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    slope[~np.isfinite(height)] = np.nan
    return slope.astype("float32")


def load_terrain_product(stem_or_dir, site_key: Optional[str] = None) -> Optional[dict]:
    """Load a datakit terrain product (<name>_terrain.npz + matching JSON)."""
    p = Path(stem_or_dir)
    if p.is_dir():
        key = (site_key or "").lower()
        candidates = sorted(p.glob(f"{key}_terrain.npz")) if key else sorted(p.glob("*_terrain.npz"))
        if not candidates:
            return None
        npz_path = candidates[0]
    else:
        npz_path = p if p.suffix == ".npz" else Path(str(p) + ".npz")
    js_path = npz_path.with_suffix(".json")
    if not (npz_path.exists() and js_path.exists()):
        return None

    with np.load(npz_path, allow_pickle=False) as data:
        if "height_m" not in data.files:
            raise ValueError(f"{npz_path}: missing required height_m array")
        height = data["height_m"].astype("float32")
        valid = data["valid_mask"].astype(bool) if "valid_mask" in data.files else np.isfinite(height)
        if valid.shape != height.shape:
            raise ValueError(f"{npz_path}: valid_mask shape does not match height_m")
        valid &= np.isfinite(height)
        slope = data["slope_deg"].astype("float32") if "slope_deg" in data.files else None
        quality_layers = {
            key: data[key].astype("float32")
            for key in (
                "aspect_deg", "roughness_m", "count",
                "elevation_uncertainty_m", "slope_uncertainty_deg",
                "slope_check_deg", "slope_check_difference_deg"
            )
            if key in data.files
        }

    if height.ndim != 2 or min(height.shape) < 2:
        raise ValueError(f"{npz_path}: height_m must be a 2-D grid at least 2x2")
    if slope is not None and slope.shape != height.shape:
        raise ValueError(f"{npz_path}: slope_deg shape does not match height_m")
    height[~valid] = np.nan
    if slope is None:
        meta = json.loads(js_path.read_text())
        mpp = float(meta.get("metres_per_pixel", 10.0))
        slope = derive_slope(height, mpp)
    else:
        slope[~valid | ~np.isfinite(slope)] = np.nan
    for key, layer in quality_layers.items():
        if layer.shape != height.shape:
            raise ValueError(f"{npz_path}: {key} shape does not match height_m")
        layer[~valid | ~np.isfinite(layer)] = np.nan
    meta = json.loads(js_path.read_text())
    return {"height": height, "slope": slope, "valid_mask": valid,
            **quality_layers, "meta": meta, "path": str(npz_path)}


def slope_stats(slope: np.ndarray, thresholds=(5, 10, 15, 20), mask: Optional[np.ndarray] = None) -> dict:
    s = np.asarray(slope, dtype=np.float64)
    valid = np.isfinite(s)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    values = s[valid]
    out = {
        "mean_slope_deg": float(np.mean(values)) if values.size else float("nan"),
        "median_slope_deg": float(np.median(values)) if values.size else float("nan"),
        "p90_slope_deg": float(np.percentile(values, 90)) if values.size else float("nan"),
        "max_slope_deg": float(np.max(values)) if values.size else float("nan"),
        "relief_proxy_m": float("nan"),
        "valid_slope_pixels": int(values.size),
    }
    for t in thresholds:
        out[f"pct_slope_le_{t}deg"] = float(100.0 * np.mean(values <= t)) if values.size else float("nan")
    return out


def largest_flat_circle_diameter_m(
    slope: np.ndarray,
    mpp: float,
    max_slope_deg: float = 10.0,
    eligible_mask: Optional[np.ndarray] = None,
) -> float:
    """Diameter of the largest circle wholly inside valid, gentle terrain."""
    ok = np.isfinite(slope) & (np.asarray(slope) <= max_slope_deg)
    if eligible_mask is not None:
        ok &= np.asarray(eligible_mask, dtype=bool)
    if not ok.any():
        return 0.0
    try:
        from scipy import ndimage
        # sampling is in metres; distance is to the nearest ineligible cell.
        dist = ndimage.distance_transform_edt(ok, sampling=float(mpp))
        return float(2.0 * dist.max())
    except ImportError:
        # Conservative fallback: a circle cannot exceed the smaller run dimension.
        return 0.0


def assess_terrain_grid(
    terrain: dict,
    radius_m: Optional[float] = None,
    slope_limit_deg: float = 10.0,
) -> dict:
    """Screen a terrain grid without treating nodata as terrain."""
    h = np.asarray(terrain["height"], dtype=np.float64)
    s = np.asarray(terrain["slope"], dtype=np.float64)
    meta = terrain["meta"]
    mpp = float(meta.get("metres_per_pixel", 10.0))
    if mpp <= 0:
        raise ValueError("metres_per_pixel must be positive")
    hvalid = np.isfinite(h)
    svalid = np.isfinite(s)
    rows, cols = h.shape
    centre = meta.get("site_pixel", [cols / 2.0, rows / 2.0])
    cx, cy = float(centre[0]), float(centre[1])
    yy, xx = np.indices(h.shape)
    dist = np.hypot((xx - cx) * mpp, (yy - cy) * mpp)
    rmax = float(radius_m if radius_m is not None else meta.get("footprint_radius_m", min(rows, cols) * mpp / 2.0))
    footprint = dist <= rmax
    inside_h = footprint & hvalid
    inside_s = footprint & svalid
    n_total = int(np.count_nonzero(footprint))
    n_valid = int(inside_h.sum())
    coverage = 100.0 * n_valid / n_total if n_total else 0.0

    stats = slope_stats(s, mask=inside_s)
    hv = h[inside_h]
    stats["relief_m"] = float(np.max(hv) - np.min(hv)) if hv.size else float("nan")
    stats["site_slope_deg"] = float(s[int(round(cy)), int(round(cx))]) if (
        0 <= int(round(cy)) < rows and 0 <= int(round(cx)) < cols
    ) else float("nan")
    stats["site_elevation_m"] = float(meta.get("site_elevation_m", float("nan")))
    stats["flat_circle_le_5deg_m"] = largest_flat_circle_diameter_m(s, mpp, 5.0, inside_h & svalid)
    stats["flat_circle_le_10deg_m"] = largest_flat_circle_diameter_m(s, mpp, 10.0, inside_h & svalid)
    stats["radius_m"] = rmax
    stats["metres_per_pixel"] = mpp
    stats["coverage_pct"] = coverage
    stats["valid_area_km2"] = n_valid * (mpp ** 2) / 1e6
    stats["valid_pixels"] = n_valid
    stats["valid_slope_pixels"] = int(inside_s.sum())
    stats["total_footprint_pixels"] = n_total
    stats["slope_limit_deg"] = float(slope_limit_deg)
    stats["passes_slope_limit"] = bool(np.isfinite(stats["site_slope_deg"]) and stats["site_slope_deg"] <= slope_limit_deg)
    stats["source"] = "dem_derived"
    stats["dem_file"] = meta.get("dem_file", "unknown")
    stats["source_metres_per_pixel"] = float(meta.get("source_metres_per_pixel", mpp))
    for layer_key, prefix in (
        ("count", "count"),
        ("elevation_uncertainty_m", "elevation_uncertainty_m"),
        ("slope_uncertainty_deg", "slope_uncertainty_deg"),
        ("roughness_m", "roughness_m"),
    ):
        layer = terrain.get(layer_key)
        if layer is not None:
            values = np.asarray(layer, dtype=float)[footprint & np.isfinite(layer)]
            if values.size:
                stats[f"{prefix}_median"] = float(np.median(values))
                stats[f"{prefix}_p90"] = float(np.percentile(values, 90))
                stats[f"{prefix}_mean"] = float(np.mean(values))
    slope_difference = terrain.get("slope_check_difference_deg")
    if slope_difference is not None:
        diff = np.abs(np.asarray(slope_difference, dtype=float))
        values = diff[footprint & np.isfinite(diff)]
        if values.size:
            stats["slope_check_mean_abs_difference_deg"] = float(np.mean(values))
            stats["slope_check_median_abs_difference_deg"] = float(np.median(values))
            stats["slope_check_p90_abs_difference_deg"] = float(np.percentile(values, 90))
            stats["slope_check_pixels"] = int(values.size)
    aspect = terrain.get("aspect_deg")
    if aspect is not None:
        av = np.asarray(aspect, dtype=float)
        av = av[footprint & np.isfinite(av)]
        if av.size:
            mean_sin = float(np.mean(np.sin(np.radians(av))))
            mean_cos = float(np.mean(np.cos(np.radians(av))))
            stats["mean_aspect_deg"] = float((np.degrees(np.arctan2(mean_sin, mean_cos)) + 360.0) % 360.0)
            stats["aspect_samples"] = int(av.size)
    stats["data_quality"] = "complete" if coverage >= 99.999 else ("partial" if coverage > 0 else "no_valid_data")

    # Retain a clearly provisional comparative index; do not present it as a
    # certified landing-safety probability or engineering acceptance score.
    gentle = stats.get("pct_slope_le_10deg", float("nan"))
    pad = min(stats["flat_circle_le_10deg_m"] / 50.0, 1.0) * 100.0
    relief = stats["relief_m"]
    relief_pen = min(max(relief, 0.0) / 200.0, 1.0) * 40.0 if np.isfinite(relief) else float("nan")
    stats["terrain_score"] = (
        round(0.5 * gentle + 0.4 * pad - 0.3 * relief_pen + 20.0, 1)
        if np.isfinite(gentle) and np.isfinite(relief) else float("nan")
    )
    stats["terrain_score_status"] = "provisional screening index; not a landing-safety certification"
    return stats


def ingest_terrain_summary_csv(path_or_buf) -> pd.DataFrame:
    """Load any Model-3-style site comparison table with loose column names."""
    df = read_table(path_or_buf)
    return rename_to_canonical(df, ["site_id", "site_name"], required=["site_id"])


def rank_sites(summary: pd.DataFrame, score_col: str = "terrain_score") -> pd.DataFrame:
    out = summary.copy()
    if score_col not in out.columns:
        return out
    out = out.sort_values(score_col, ascending=False, na_position="last").reset_index(drop=True)
    out["terrain_rank"] = np.arange(1, len(out) + 1)
    return out
