import csv
import json
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python tools/validate_terrain_csv.py SITES.csv TERRAIN_DIR"
    )

sites_file = Path(sys.argv[1])
terrain_dir = Path(sys.argv[2])

with sites_file.open(newline="") as f:
    sites = list(csv.DictReader(f))

failures = []

for site in sites:
    sid = site["site_id"]
    key = sid.lower()

    npz = terrain_dir / f"{key}_terrain.npz"
    meta_file = terrain_dir / f"{key}_terrain.json"
    provenance = terrain_dir / f"{key}_terrain.npz.provenance.json"

    print(f"\n=== {sid} ===")

    for path in (npz, meta_file, provenance):
        ok = path.is_file()
        print(f"[{'PASS' if ok else 'FAIL'}] {path.name}")
        if not ok:
            failures.append(f"{sid}: missing {path.name}")

    if not (npz.exists() and meta_file.exists()):
        continue

    try:
        meta = json.loads(meta_file.read_text())
        with np.load(npz) as data:
            required = {"height_m", "slope_deg", "valid_mask"}
            missing = required - set(data.files)

            if missing:
                raise ValueError(f"missing arrays: {sorted(missing)}")

            height = data["height_m"]
            slope = data["slope_deg"]
            mask = data["valid_mask"].astype(bool)

            if height.ndim != 2 or height.shape != slope.shape:
                raise ValueError("height/slope shapes invalid")

            if mask.shape != height.shape:
                raise ValueError("valid mask shape invalid")

            if not mask.any():
                raise ValueError("no valid pixels")

            if not np.isfinite(height[mask]).all():
                raise ValueError("non-finite valid elevation values")

            if not np.isfinite(slope[mask]).all():
                raise ValueError("non-finite valid slope values")

            coverage = float(meta.get("grid_coverage_pct", 0))
            source_mpp = float(meta.get("source_metres_per_pixel", 0))

            if coverage < 99.999:
                raise ValueError(f"footprint coverage only {coverage:.3f}%")

            if source_mpp <= 0:
                raise ValueError("invalid source resolution metadata")

            print(f"[PASS] grid shape: {height.shape}")
            print(f"[PASS] valid pixels: {mask.sum():,}")
            print(f"[PASS] coverage: {coverage:.3f}%")
            print(f"[PASS] native source resolution: {source_mpp:g} m/pixel")
            print(f"[PASS] DEM: {meta.get('dem_file', 'unknown')}")

    except Exception as e:
        failures.append(f"{sid}: {e}")
        print(f"[FAIL] {e}")

print("\n================================")
print(f"Sites checked: {len(sites)}")
print(f"Failures: {len(failures)}")

for failure in failures:
    print(" -", failure)

if failures:
    raise SystemExit(1)

print("ALL TERRAIN PRODUCTS PASSED")
