# Model 3 — real DEM integration

This workflow keeps the existing Streamlit UI and model interfaces. Raw rasters remain outside the repository; Model 3 consumes small, traceable terrain products generated from them.

## What the processor does

- Reads DEMs in raster windows; it does not load multi-gigabyte rasters into memory.
- For each candidate site, chooses the finest DEM that covers the complete requested grid footprint. A coarser regional DEM is selected if a high-resolution tile does not cover that whole footprint.
- Applies raster/GDAL scale and offset metadata, or PDS labels where available. Missing scaling metadata is reported loudly.
- Calculates slope at the source DEM resolution, then downsamples only for the saved display grid.
- Derives downslope aspect clockwise from local true north and local elevation standard-deviation roughness.
- Preserves nodata instead of filling it with mean elevation.
- Optionally aligns LOLA count, elevation uncertainty, slope uncertainty, and an independent slope raster onto the DEM grid. A supplied slope is used as a cross-check; the DEM-derived slope remains the screening layer.
- Writes source fingerprints and processing parameters beside every derived terrain product.

## Prepare

From the project directory:

```bash
pip install -r requirements.txt
pip install -r requirements-geo.txt
```

The project archive does not contain the multi-gigabyte source rasters. The helper script uses the existing local dataset folder and skips any optional file that is not present; it does not download anything.

Run the full workflow with:

```bash
bash tools/run_model3_local.sh
```

Set `CLPS_DATA_DIR=/path/to/your/datasets` before running if your raster folder is not `~/Desktop/clpsdatasets`.

The processor stops if a DEM or companion layer has no documented scale/offset. Inspect it with `gdalinfo` and its PDS label first. Only if you verify that its stored values are already in the declared units (DEM metres, count values, uncertainty metres/degrees) should you run:

```bash
MODEL3_ALLOW_ASSUMED_SCALING=1 bash tools/run_model3_local.sh
```

Do not use that override merely to bypass the warning; a wrong scale corrupts elevation, slope and suitability metrics.

## Build all four site grids manually

```bash
cd ~/Desktop/clps/clpsnew

D="$HOME/Desktop/clpsdatasets"
DEM_ARGS=()
COUNT_ARGS=()
ELEV_UNC_ARGS=()
SLOPE_UNC_ARGS=()
SLOPE_CHECK_ARGS=()

# Regional coverage plus available local high-resolution DEMs.
for f in \
  "LDEM_80S_20MPP_ADJ.tiff" \
  "LDEM_83S_10MPP_ADJ.tiff" \
  "ldem_87s_5mpp.tif" \
  "Site01_final_adj_5mpp_surf.tif" \
  "Site06_final_adj_5mpp_surf.tif" \
  "NAC_DTM_NOBILE02.TIF" \
  "NAC_DTM_NOBILE03.TIF"
do
  if [ -f "$D/$f" ]; then
    DEM_ARGS+=(--dem "$D/$f")
    echo "DEM: $f"
  else
    echo "Not present, skipping DEM: $f"
  fi
done

# LOLA count/observation support layers.
for f in \
  "LDEC_80S_20MPP_ADJ.tiff" \
  "LDEC_83S_10MPP_ADJ.TIF" \
  "Site01_final_adj_5mpp_ldec.tif" \
  "Site06_final_adj_5mpp_ldec.tif"
do
  if [ -f "$D/$f" ]; then COUNT_ARGS+=(--count "$D/$f"); fi
done

# Elevation and slope uncertainty products, where available.
for f in \
  "Site01_final_adj_5mpp_toterr.tif" \
  "Site06_final_adj_5mpp_toterr.tif"
do
  if [ -f "$D/$f" ]; then ELEV_UNC_ARGS+=(--elev-uncertainty "$D/$f"); fi
done

for f in \
  "Site01_final_adj_5mpp_slperr.tif" \
  "Site06_final_adj_5mpp_slperr.tif"
do
  if [ -f "$D/$f" ]; then SLOPE_UNC_ARGS+=(--slope-uncertainty "$D/$f"); fi
done

# Independent slope products validate the slope derived from each DEM.
for f in \
  "ldsm_87s_5mpp.tif" \
  "Site01_final_adj_5mpp_slp.tif" \
  "Site06_final_adj_5mpp_slp.tif"
do
  if [ -f "$D/$f" ]; then SLOPE_CHECK_ARGS+=(--slope-check "$D/$f"); fi
done

if [ "${#DEM_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: no candidate DEMs found in $D"
  exit 1
fi

python tools/datakit.py dem \
  "${DEM_ARGS[@]}" \
  --sites data/sample_sites.csv \
  --products terrain \
  --radius 2500 \
  --footprint-radius 1000 \
  --max-grid 512 \
  "${COUNT_ARGS[@]}" \
  "${ELEV_UNC_ARGS[@]}" \
  "${SLOPE_UNC_ARGS[@]}" \
  "${SLOPE_CHECK_ARGS[@]}" \
  --slope-check-units degrees \
  --out derived/model3 \
  --allow-large
```

`--radius 1000` creates a 2 km-wide grid matching the 1 km-radius landing-screening footprint. This allows local 4–5 m tiles to qualify when they cover the complete analysis area; sites without such coverage fall back to the best complete-coverage regional DEM. Increase `--radius` only when you deliberately want a larger context grid and accept that smaller local tiles may no longer qualify. The best available source resolution is retained in metadata; the display grid is downsampled only if it exceeds 512 pixels per side.

The 87S hillshade is not needed for Model 3 calculations: the app derives hillshade from the DEM. The 87S and site-specific slope products are used as independent validation layers instead.

## Install the products into the app

```bash
cp -v derived/model3/terrain/* data/terrain/

python tools/build_model3_comparison.py

python tests/validate_project.py
```

The comparison builder refuses to overwrite the existing Model 3 CSV if even one site is missing a processed terrain product. It then refreshes the existing four-panel slope image from the new per-site grids. No UI layout or Model 1/Model 2 code is changed.

## Interpretation

- `height_m`: elevation relative to the candidate site's DEM elevation.
- `slope_deg`: DEM-derived slope in degrees.
- `aspect_deg`: downslope azimuth clockwise from local true north; undefined on nearly flat cells.
- `roughness_m`: local elevation standard deviation, with the physical window recorded in the JSON metadata.
- `count`: aligned observation-count support, where a matching product exists.
- `elevation_uncertainty_m` / `slope_uncertainty_deg`: aligned source uncertainty layers, where available.
- `slope_check_difference_deg`: independent supplied slope minus DEM-derived slope.
- `coverage_pct`: valid elevation coverage inside the 1 km analysis footprint.
- `terrain_score_provisional`: comparative screening index only, not a certified landing-safety probability.

A site without a local high-resolution tile may legitimately use the regional DEM. The report records the actual source and resolution for each site; it must not imply that every site has 4 m or 5 m coverage.
