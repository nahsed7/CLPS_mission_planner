#!/usr/bin/env python3
"""Rebuild the packaged Model 3 comparison CSV from processed, real DEM grids.

Refuses to mix fresh DEM-derived metrics with legacy summary rows. Every site
in data/sample_sites.csv must have a corresponding data/terrain/*_terrain.npz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clps_engine.terrain import assess_terrain_grid, load_terrain_product


def largest_patch_area(slope, mpp, threshold, footprint):
    ok = np.isfinite(slope) & (slope <= threshold) & footprint
    if not ok.any():
        return 0.0
    labels, n = ndimage.label(ok, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return 0.0
    counts = np.bincount(labels.ravel())
    return float(counts[1:].max(initial=0) * mpp * mpp)



def write_slope_map(products, sites, output):
    """Refresh the existing four-panel Model 3 slope image from the real grids."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print("matplotlib not installed; comparison CSV built, existing slope map left unchanged.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    axes = axes.ravel()
    image = None
    for ax, (_, site) in zip(axes, sites.iterrows()):
        sid = str(site["site_id"])
        product = products[sid]
        meta = product["meta"]
        slope = np.ma.masked_invalid(product["slope"])
        rows, cols = slope.shape
        cx, cy = meta.get("site_pixel", [cols / 2, rows / 2])
        mppx, mppy = meta.get("metres_per_pixel_xy", [
            meta.get("metres_per_pixel", 10.0),
            meta.get("metres_per_pixel", 10.0),
        ])
        lon = np.radians(float(meta.get("lon", site["longitude_deg"])))
        east_xy = np.asarray(meta.get("east_xy", [np.cos(lon), -np.sin(lon)]), dtype=float)
        north_xy = np.asarray(meta.get("north_xy", [np.sin(lon), np.cos(lon)]), dtype=float)
        basis = np.column_stack((east_xy, north_xy))
        inv_basis = np.linalg.inv(basis)
        xmap = (np.arange(cols)[None, :] - cx) * float(mppx)
        ymap = (cy - np.arange(rows)[:, None]) * float(mppy)
        east = (inv_basis[0, 0] * xmap + inv_basis[0, 1] * ymap) / 1000.0
        north = (inv_basis[1, 0] * xmap + inv_basis[1, 1] * ymap) / 1000.0
        image = ax.pcolormesh(east, north, slope, shading="auto",
                              cmap="RdYlGn_r", vmin=0, vmax=30)
        footprint_km = float(meta.get("footprint_radius_m", 1000.0)) / 1000.0
        grid_half_width = float(meta.get("grid_half_width_m", min(rows, cols) * float(meta.get("metres_per_pixel", 10.0)) / 2.0))
        grid_km = grid_half_width / 1000.0
        ax.add_patch(Circle((0, 0), footprint_km, fill=False, edgecolor="white", linewidth=1.8))
        ax.add_patch(Circle((0, 0), grid_km, fill=False, edgecolor="white",
                            linewidth=0.8, linestyle="--", alpha=0.6))
        ax.plot(0, 0, marker="+", color="black", markersize=10, markeredgewidth=2)
        coverage = product["stats"]["coverage_pct"]
        resolution = meta.get("source_metres_per_pixel", meta.get("metres_per_pixel", np.nan))
        ax.set_title(f"{site.get('site_name', sid)} · {resolution:g} m source")
        ax.set_xlabel("km east of site")
        ax.set_ylabel("km north of site")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-grid_km, grid_km)
        ax.set_ylim(-grid_km, grid_km)
        ax.text(0.02, 0.02, f"valid coverage {coverage:.1f}%",
                transform=ax.transAxes, color="white", fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 3})
    fig.suptitle("DEM-derived slope screening · white circle = analysis footprint · not a landing-safety assessment")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.78, label="Slope (degrees)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Updated slope map: {output}")

def build(input_sites: Path, terrain_dir: Path, output: Path, slope_map_output: Path):
    sites = pd.read_csv(input_sites)
    rows, missing = [], []
    products = {}
    for _, site in sites.iterrows():
        sid = str(site["site_id"])
        product = load_terrain_product(terrain_dir, sid.lower())
        if product is None:
            missing.append(sid)
            continue
        meta = product["meta"]
        radius = float(meta.get("footprint_radius_m", 1000.0))
        stats = assess_terrain_grid(product, radius_m=radius)
        product["stats"] = stats
        products[sid] = product
        h, slope = product["height"], product["slope"]
        mpp = float(meta.get("metres_per_pixel", 10.0))
        cx, cy = meta.get("site_pixel", [h.shape[1] / 2, h.shape[0] / 2])
        yy, xx = np.indices(h.shape)
        footprint = np.hypot((xx - cx) * mpp, (yy - cy) * mpp) <= radius
        below5 = stats.get("pct_slope_le_5deg", float("nan"))
        below10 = stats.get("pct_slope_le_10deg", float("nan"))
        below15 = stats.get("pct_slope_le_15deg", float("nan"))
        row = {
            "site_id": sid,
            "site_name": str(site.get("site_name", sid)),
            "lat_deg": float(site["latitude_deg"]),
            "lon_deg": float(site["longitude_deg"]),
            "footprint_radius_m": radius,
            "cell_size_m": mpp,
            "source_cell_size_m": float(meta.get("source_metres_per_pixel", mpp)),
            "n_valid_pixels": stats["valid_pixels"],
            "valid_area_km2": stats["valid_area_km2"],
            "coverage_pct": stats["coverage_pct"],
            "mean_slope_deg": stats["mean_slope_deg"],
            "median_slope_deg": stats["median_slope_deg"],
            "p90_slope_deg": stats["p90_slope_deg"],
            "pct_slope_le_5deg": below5,
            "pct_slope_le_10deg": below10,
            "pct_slope_gt_10deg": 100.0 - below10 if np.isfinite(below10) else np.nan,
            "pct_slope_gt_15deg": 100.0 - below15 if np.isfinite(below15) else np.nan,
            "largest_flat_patch_le_5deg_m2": largest_patch_area(slope, mpp, 5.0, footprint),
            "largest_flat_circle_le_5deg_diameter_m": stats["flat_circle_le_5deg_m"],
            "largest_flat_patch_le_10deg_m2": largest_patch_area(slope, mpp, 10.0, footprint),
            "largest_flat_circle_le_10deg_diameter_m": stats["flat_circle_le_10deg_m"],
            "relief_m": stats["relief_m"],
            "site_elevation_m": stats["site_elevation_m"],
            "site_slope_deg": stats["site_slope_deg"],
            "roughness_median_m": stats.get("roughness_m_median", np.nan),
            "roughness_p90_m": stats.get("roughness_m_p90", np.nan),
            "mean_aspect_deg": stats.get("mean_aspect_deg", np.nan),
            "count_median": stats.get("count_median", np.nan),
            "elevation_uncertainty_m_median": stats.get("elevation_uncertainty_m_median", np.nan),
            "slope_uncertainty_deg_median": stats.get("slope_uncertainty_deg_median", np.nan),
            "slope_check_mean_abs_difference_deg": stats.get("slope_check_mean_abs_difference_deg", np.nan),
            "terrain_score_provisional": stats["terrain_score"],
            "terrain_rank_provisional": np.nan,
            "terrain_source": meta.get("dem_file", "unknown"),
            "data_quality": stats["data_quality"],
            "notes": "DEM-derived; score is a provisional screening index, not landing-safety certification",
        }
        rows.append(row)

    if missing:
        raise SystemExit(
            "Refusing to mix new DEM-derived rows with legacy CSV values. "
            "Missing terrain products for: " + ", ".join(missing) +
            ". Generate/copy every site's terrain NPZ+JSON first."
        )
    result = pd.DataFrame(rows).sort_values(
        "terrain_score_provisional", ascending=False, na_position="last"
    ).reset_index(drop=True)
    result["terrain_rank_provisional"] = np.arange(1, len(result) + 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    result.to_csv(temp, index=False)
    temp.replace(output)
    write_slope_map(products, sites, slope_map_output)
    print(f"Wrote {output}")
    print(result[[
        "site_id", "mean_slope_deg", "pct_slope_le_10deg",
        "largest_flat_circle_le_10deg_diameter_m", "relief_m",
        "coverage_pct", "terrain_score_provisional", "terrain_rank_provisional",
        "terrain_source",
    ]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=Path, default=ROOT / "data/sample_sites.csv")
    parser.add_argument("--terrain-dir", type=Path, default=ROOT / "data/terrain")
    parser.add_argument("--output", type=Path, default=ROOT / "data/model3_site_comparison.csv")
    parser.add_argument("--slope-map-output", type=Path, default=ROOT / "static/slope_maps_all_sites.png")
    args = parser.parse_args()
    build(args.sites, args.terrain_dir, args.output, args.slope_map_output)


if __name__ == "__main__":
    main()
