# CLPS New — Lunar Site Evaluation Device

NASA Space Apps Challenge 2026. A **processable** engine (not a fixed four-site report) that turns any sites / horizon / geometry / terrain dataset into:

- **Solar availability** windows (Sun vs terrain horizon)
- **Direct-to-Earth (DTE)** communication windows
- **Power generation** (array W, energy, battery SoC, eclipse survival)
- **Terrain screening** from datakit products

Saved separately under `Desktop/clps/clpsnew/` — original `lunar-mission-browser-main` is untouched.

## Why this exists

The previous stack (~60%) had a good Streamlit browser and hard-coded Model 1–3 scripts that emitted CSVs for four sites. Gaps vs the challenge:

| Challenge need | Before | CLPS New |
|---|---|---|
| Rapid site + date comparison | Fixed SITES dict | Any CSV + custom lat/lon |
| Power generation potential | Binary sunlit only | Full power module |
| Process given datasets | Report scripts | `MissionPipeline` feeds |
| Time to sunset / Earthset | Partial | Explicit operational clocks |
| Works with new data | Re-edit scripts | Upload / CLI / datakit |

## Quick start

```bash
cd ~/Desktop/clps/clpsnew
pip install -r requirements.txt
streamlit run app.py
```

CLI (no UI):

```bash
python run_pipeline.py --sites data/sample_sites.csv \
  --start 2026-10-15T00:00:00 --hours 336 --step 60 \
  --out outputs/demo
```

## Feed it data

```python
from clps_engine import MissionPipeline, load_sites_csv

pipe = MissionPipeline()
pipe.feed_horizons("my_horizons.csv")          # optional
pipe.feed_geometry("spice_geometry.csv")       # optional — skips live ephemeris
sites = load_sites_csv("any_sites.csv")
results = pipe.process_many(sites, start_iso="2026-10-15T00:00:00", hours=336)
pipe.save_comparison(results, "outputs/run")
```

Column names are matched loosely — see `data/schemas.md`.

## Layout

```
clpsnew/
  app.py                 # Streamlit device UI
  run_pipeline.py        # CLI
  clps_engine/
    adapters.py          # Loose CSV schema matching
    sites.py             # Curated + custom + CSV sites
    geometry.py          # Skyfield live OR CSV/NPZ ingest
    horizon.py           # Terrain masks
    solar.py             # Model 1 generic
    dte.py               # Model 2 generic
    power.py             # NEW — power generation
    terrain.py           # Model 3 generic
    windows.py           # Runs / time-to-set
    pipeline.py          # Orchestrator device
  tools/datakit.py       # Extract small products from huge DEM/SPICE
  data/                  # Sample sites, horizons, terrain
  outputs/               # Written results (git-ignore friendly)
```

## Building inputs with datakit

```bash
# Model 3: provide all candidate DEMs. For each site, datakit chooses
# the finest raster that covers the COMPLETE requested terrain footprint.
# Terrain grids preserve nodata and derive slope at the source DEM resolution.
python tools/datakit.py dem --dem fine_site_dem.tif --dem regional_5m_dem.tif \
  --dem south_pole_20m_dem.tif --sites data/sample_sites.csv \
  --products terrain --radius 1000 --footprint-radius 1000 \
  --max-grid 512 --out derived/model3/

# Optional quality rasters: use matching count / uncertainty layers and
# an independent slope product to validate the DEM-derived slope.
python tools/datakit.py dem --dem best_dem.tif --sites data/sample_sites.csv \
  --products terrain --radius 1000 \
  --count ldec_count.tif \
  --elev-uncertainty total_elevation_error.tif \
  --slope-uncertainty slope_error.tif \
  --slope-check supplied_slope.tif --slope-check-units degrees \
  --out derived/model3/

# Horizon products use ordered distance bands (finest/near to coarsest/far).
python tools/datakit.py dem --dem near.jp2:10000 --dem far.jp2:70000 \
  --sites data/sample_sites.csv --products horizon --out derived/horizons/

# Ephemeris tables (no kernels needed at runtime afterwards)
python tools/datakit.py spice --kernels /path/to/kernels \
  --start 2026-10-15T00:00:00 --days 14 --sites data/sample_sites.csv --out derived/
```

Copy `horizon_profiles.csv` into `data/` and each site's `*_terrain.npz/json` into `data/terrain/`.
Then rebuild the four-site Model 3 comparison from those grids:

```bash
python tools/build_model3_comparison.py
```

The comparison builder refuses to mix new DEM-derived metrics with stale legacy rows if any site is missing a terrain grid. The geospatial CLI dependencies are listed in `requirements-geo.txt`. Raw multi-GB rasters stay outside the app repository; derived terrain grids carry per-file provenance.

## Power model (brief)

\(P = P_\mathrm{nameplate} \cdot \mu \cdot f_\mathrm{dust} \cdot f_\mathrm{temp}\) while sunlit, where \(\mu\) is the cosine of incidence for `fixed` / `vertical_axis` / `dual_axis` tracking. Battery SoC is a simple coulomb counter against a continuous load — enough for rapid feasibility, not a full EPS design.
