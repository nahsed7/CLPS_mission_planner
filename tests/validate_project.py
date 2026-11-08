#!/usr/bin/env python3
"""Project validation — runs against packaged CSVs (no SPICE/LOLA required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from clps_engine.offline import (
    load_dte_summary,
    load_dte_timeline,
    load_model3,
    load_solar_summary,
    load_solar_timeline,
    merge_offline_geometry,
    offline_team_scorecard,
)
from clps_engine.pipeline import MissionPipeline
from clps_engine.power import ArrayConfig
from clps_engine.sites import curated_clps_sites
from clps_engine.terrain import assess_terrain_grid, derive_slope, load_terrain_product
from clps_engine.windows import longest_run_hours


def approx(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return bool(cond)


def main():
    print("CLPS New validation suite\n")
    fails = 0

    print("1. Packaged team outputs present")
    for p in [
        ROOT / "data/solar/solar_site_comparison.csv",
        ROOT / "data/solar/solar_timeline_all_sites.csv",
        ROOT / "data/dte/four_site_dte_summary.csv",
        ROOT / "data/dte/LM1_dte_timeline.csv",
        ROOT / "data/model3_site_comparison.csv",
        ROOT / "data/horizons/lm1_horizon.csv",
        ROOT / "data/terrain/lm1_terrain.npz",
    ]:
        fails += not check(str(p.relative_to(ROOT)), p.exists())

    print("\n2. Model 1 summary matches timeline recomputation")
    solar_sum = load_solar_summary()
    for _, row in solar_sum.iterrows():
        sid = row["site_id"]
        tl = load_solar_timeline(sid)
        pct = 100.0 * tl["sunlit"].mean()
        ok = approx(pct, row["sunlight_percent"], 0.05)
        fails += not check(f"{sid} sunlit %", ok, f"timeline={pct:.4f} summary={row['sunlight_percent']}")
        step = 10.0
        dark = longest_run_hours(~tl["sunlit"].to_numpy(), step)
        fails += not check(
            f"{sid} longest darkness",
            approx(dark, row["longest_darkness_hours"], 0.2),
            f"recalc={dark:.3f} summary={row['longest_darkness_hours']}",
        )

    print("\n3. Model 2 summary matches timeline recomputation")
    dte_sum = load_dte_summary()
    for _, row in dte_sum.iterrows():
        sid = row["site_id"]
        tl = load_dte_timeline(sid)
        pct = 100.0 * tl["earth_visible"].mean()
        fails += not check(
            f"{sid} Earth %",
            approx(pct, row["visibility_percent"], 0.05),
            f"timeline={pct:.4f} summary={row['visibility_percent']}",
        )

    print("\n4. Model 3 schema + ranks")
    m3 = load_model3()
    fails += not check("4 sites in Model 3", len(m3) == 4)
    fails += not check("ranks 1..4", set(m3["terrain_rank_provisional"].astype(int)) == {1, 2, 3, 4})

    print("\n5. Model 3 real-grid handling")
    lm1_grid = load_terrain_product(ROOT / "data/terrain", "lm1")
    lm1_metrics = assess_terrain_grid(lm1_grid, radius_m=1000)
    fails += not check("LM1 DEM grid loads", lm1_grid is not None)
    fails += not check("LM1 footprint coverage is measured",
                       approx(lm1_metrics["coverage_pct"], 100.0, 0.001),
                       f"coverage={lm1_metrics['coverage_pct']:.3f}%")
    yy, xx = np.mgrid[0:20, 0:20]
    synthetic = (0.1 * xx * 10 + 0.05 * yy * 10).astype(float)
    synthetic[8:11, 8:11] = np.nan
    derived_slope = derive_slope(synthetic, 10, 10)
    expected_slope = np.degrees(np.arctan(np.hypot(0.1, 0.05)))
    fails += not check("slope derivative matches analytic plane",
                       approx(np.nanmedian(derived_slope), expected_slope, 1e-4))
    fails += not check("nodata remains nodata", np.isnan(derived_slope[9, 9]))

    print("\n6. Offline pipeline + power + RF")
    sites = [s for s in curated_clps_sites() if s.id in ("LM1", "Site06", "Site20", "Site23")]
    pipe = MissionPipeline(mode="offline", array=ArrayConfig(nameplate_w=500, tracking="dual_axis", night_draw_w=40, battery_wh=3000))
    results = pipe.process_many(sites, mode="offline")
    fails += not check("processed 4 sites", len(results) == 4)
    table = pipe.comparison_table(results)
    fails += not check("power columns present", "power_mean_w" in table.columns and table["power_mean_w"].notna().all())
    fails += not check("rf columns present", "rf_geometry_ok_pct" in table.columns)
    legacy_terrain = pipe.resolve_terrain(sites[1])
    fails += not check("legacy mean slope not mislabeled as site slope",
                       legacy_terrain.get("source") != "model3_csv" or "site_slope_deg" not in legacy_terrain)
    out = ROOT / "outputs" / "validation"
    pipe.save_comparison(results, out)
    fails += not check("wrote comparison", (out / "site_comparison.csv").exists())

    print("\n7. Horizon mask for LM1 loads")
    mask = pipe.resolve_horizon(sites[0])
    fails += not check("LM1 horizon mask", mask is not None and len(mask[0]) >= 8)

    print("\n8. Team scorecard builds")
    sc = offline_team_scorecard()
    fails += not check("scorecard 4 rows", len(sc) == 4)
    sc.to_csv(out / "team_scorecard.csv", index=False)

    print("\n9. Offline geometry merge length")
    g = merge_offline_geometry(sites[0])
    fails += not check("LM1 merged samples", len(g) == 12824, f"n={len(g)}")

    report = {"fails": fails, "status": "PASS" if fails == 0 else "FAIL"}
    (out / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n=== {report['status']} ({fails} failure(s)) ===")
    return fails


if __name__ == "__main__":
    sys.exit(main())
