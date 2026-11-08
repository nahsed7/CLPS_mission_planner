#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${CLPS_DATA_DIR:-$HOME/Desktop/clpsdatasets}"
cd "$ROOT"

# Discover the actual LOLA south-polar GeoTIFFs without relying on case-sensitive
# extensions. Keep unrelated regional NAC DTMs out until their PDS4 labels have
# been checked independently.
mapfile -d '' DEM_FILES < <(find "$D" -maxdepth 1 -type f \
  \( -iname 'LDEM_80S_20MPP_ADJ.*' \
     -o -iname 'LDEM_83S_10MPP_ADJ.*' \
     -o -iname 'ldem_87s_5mpp.tif' \
     -o -iname '*_final_adj_5mpp_surf.tif' \) -print0 | sort -z)

DEM_ARGS=()
for f in "${DEM_FILES[@]}"; do
  DEM_ARGS+=(--dem "$f")
  echo "DEM: $(basename "$f")"
done

# Companion products: the PGDA product documentation defines their native
# units (counts, metres, and degrees respectively). Rasterio/GDAL may omit
# identity scale metadata, so the explicit identity handling below is used.
COUNT_ARGS=()
ELEV_UNC_ARGS=()
SLOPE_UNC_ARGS=()
SLOPE_CHECK_ARGS=()
while IFS= read -r -d '' f; do COUNT_ARGS+=(--count "$f"); echo "COUNT: $(basename "$f")"; done < <(
  find "$D" -maxdepth 1 -type f \
    \( -iname 'LDEC_80S_20MPP_ADJ.*' -o -iname 'LDEC_83S_10MPP_ADJ.*' \
       -o -iname 'ldec_87s_5mpp.tif' -o -iname '*_final_adj_5mpp_ldec.tif' \) -print0 | sort -z)
while IFS= read -r -d '' f; do ELEV_UNC_ARGS+=(--elev-uncertainty "$f"); echo "ELEV UNC: $(basename "$f")"; done < <(
  find "$D" -maxdepth 1 -type f -iname '*_final_adj_5mpp_toterr.tif' -print0 | sort -z)
while IFS= read -r -d '' f; do SLOPE_UNC_ARGS+=(--slope-uncertainty "$f"); echo "SLOPE UNC: $(basename "$f")"; done < <(
  find "$D" -maxdepth 1 -type f -iname '*_final_adj_5mpp_slperr.tif' -print0 | sort -z)
while IFS= read -r -d '' f; do SLOPE_CHECK_ARGS+=(--slope-check "$f"); echo "SLOPE CHECK: $(basename "$f")"; done < <(
  find "$D" -maxdepth 1 -type f \
    \( -iname 'ldsm_87s_5mpp.tif' -o -iname '*_final_adj_5mpp_slp.tif' \) -print0 | sort -z)

if [[ ${#DEM_ARGS[@]} -eq 0 ]]; then
  echo "ERROR: no supported LOLA DEMs found in $D" >&2
  exit 1
fi

mkdir -p derived/model3 data/terrain

# PGDA adjusted LOLA DEM GeoTIFFs document Z in metres. Explicit identity
# scaling prevents datakit from treating the documented units as a guess.
# Companion products also use their documented native units; permit missing
# identity metadata for those layers only.
python tools/datakit.py dem \
  "${DEM_ARGS[@]}" \
  --sites data/sample_sites.csv \
  --products terrain \
  --radius 1000 \
  --footprint-radius 1000 \
  --max-grid 512 \
  "${COUNT_ARGS[@]}" \
  "${ELEV_UNC_ARGS[@]}" \
  "${SLOPE_UNC_ARGS[@]}" \
  "${SLOPE_CHECK_ARGS[@]}" \
  --scale 1 --offset 0 \
  --allow-assumed-scaling \
  --slope-check-units degrees \
  --out derived/model3 \
  --allow-large

cp -v derived/model3/terrain/* data/terrain/
python tools/build_model3_comparison.py
python tests/validate_project.py

echo
echo "Model 3 terrain integration complete."
echo "Raw datasets remain in: $D"
echo "Derived products: $ROOT/data/terrain"
