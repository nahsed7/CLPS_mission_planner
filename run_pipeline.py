#!/usr/bin/env python3
"""CLI front-end for the CLPS New processing device.

Examples
--------
  # Live ephemeris for curated sites, 14 Earth-days, save outputs
  python run_pipeline.py --sites data/sample_sites.csv \\
      --start 2026-10-15T00:00:00 --hours 336 --step 60 \\
      --out outputs/run1

  # Use a precomputed geometry CSV instead of Skyfield
  python run_pipeline.py --sites data/sample_sites.csv \\
      --geometry path/to/geometry.csv --out outputs/from_spice

  # Custom single site
  python run_pipeline.py --lat -89.5 --lon 45 --name "My pad" \\
      --start 2026-10-15T00:00:00 --hours 168 --out outputs/custom
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clps_engine.pipeline import MissionPipeline
from clps_engine.power import ArrayConfig
from clps_engine.sites import curated_clps_sites, custom_site, load_sites_csv, merge_sites


def main(argv=None):
    ap = argparse.ArgumentParser(description="CLPS New — process lunar site datasets")
    ap.add_argument("--sites", help="Sites CSV (loose column names)")
    ap.add_argument("--curated", action="store_true", help="Also include curated CLPS sites")
    ap.add_argument("--lat", type=float, help="Custom latitude")
    ap.add_argument("--lon", type=float, help="Custom longitude")
    ap.add_argument("--elev", type=float, default=0.0)
    ap.add_argument("--name", default="Custom")
    ap.add_argument("--id", default="CUSTOM")
    ap.add_argument("--horizon", help="Horizon CSV to feed")
    ap.add_argument("--geometry", help="Precomputed geometry CSV")
    ap.add_argument("--geometry-npz", help="datakit ephemeris_*.npz")
    ap.add_argument("--start", default="2026-10-15T00:00:00")
    ap.add_argument("--hours", type=float, default=336.0)
    ap.add_argument("--step", type=float, default=60.0)
    ap.add_argument("--earth-rule", choices=["any_part", "centre", "whole_disc"], default="any_part")
    ap.add_argument("--nameplate-w", type=float, default=500.0)
    ap.add_argument("--tracking", choices=["fixed", "vertical_axis", "dual_axis"], default="fixed")
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--load-w", type=float, default=50.0)
    ap.add_argument("--battery-wh", type=float, default=2000.0)
    ap.add_argument("--out", default="outputs/latest")
    ap.add_argument("--site-id", action="append", help="Only process these site ids")
    args = ap.parse_args(argv)

    groups = []
    if args.curated or (not args.sites and args.lat is None):
        groups.append(curated_clps_sites())
    if args.sites:
        groups.append(load_sites_csv(args.sites))
    if args.lat is not None and args.lon is not None:
        groups.append([custom_site(args.lat, args.lon, args.elev, args.name, args.id)])
    registry = merge_sites(*groups)
    sites = list(registry.values())
    if args.site_id:
        want = set(args.site_id)
        sites = [s for s in sites if s.id in want]
        if not sites:
            raise SystemExit(f"No sites matched {want}")

    pipe = MissionPipeline(
        earth_rule=args.earth_rule,
        array=ArrayConfig(
            nameplate_w=args.nameplate_w,
            tracking=args.tracking,
            tilt_deg=args.tilt,
            night_draw_w=args.load_w,
            battery_wh=args.battery_wh,
        ),
    )
    if args.horizon:
        pipe.feed_horizons(args.horizon)
    if args.geometry:
        pipe.feed_geometry(args.geometry)
    if args.geometry_npz:
        pipe.feed_geometry_npz(args.geometry_npz)

    if args.geometry or args.geometry_npz:
        results = [pipe.process_site(s) for s in sites]
    else:
        results = pipe.process_many(sites, start_iso=args.start, hours=args.hours, step_min=args.step)

    out = Path(args.out)
    path = pipe.save_comparison(results, out)
    table = pipe.comparison_table(results)
    print(table.to_string(index=False))
    print(f"\nWrote comparison → {path}")
    print(f"Per-site timelines / windows / JSON under {out}/")
    summary = {
        "n_sites": len(results),
        "start": args.start,
        "hours": args.hours,
        "earth_rule": args.earth_rule,
        "sites": [r.site.id for r in results],
    }
    (out / "run_meta.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
