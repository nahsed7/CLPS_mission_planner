#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${CLPS_DATA_DIR:-$HOME/Desktop/clpsdatasets}"

SITES="${1:-data/research_candidate_sites.csv}"
TAG="${2:-research}"

cd "$ROOT"

if [[ ! -f "$SITES" ]]; then
    echo "ERROR: site CSV not found: $SITES"
    exit 1
fi

if [[ ! -d "$D" ]]; then
    echo "ERROR: dataset directory not found: $D"
    exit 1
fi

echo "======================================"
echo "CLPS Model 3 — Multi-site terrain run"
echo "Sites: $SITES"
echo "Datasets: $D"
echo "Output tag: $TAG"
echo "======================================"

mapfile -d '' DEM_FILES < <(
    find "$D" -maxdepth 1 -type f \
      \( -iname 'LDEM_80S_20MPP_ADJ.*' \
         -o -iname 'LDEM_83S_10MPP_ADJ.*' \
         -o -iname 'ldem_87s_5mpp.tif' \
         -o -iname '*_final_adj_5mpp_surf.tif' \) \
      -print0 | sort -z
)

DEM_ARGS=()
for f in "${DEM_FILES[@]}"; do
    DEM_ARGS+=(--dem "$f")
    echo "DEM: $(basename "$f")"
done

if [[ ${#DEM_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: no supported DEMs discovered."
    exit 1
fi

COUNT_ARGS=()
while IFS= read -r -d '' f; do
    COUNT_ARGS+=(--count "$f")
    echo "COUNT: $(basename "$f")"
done < <(
    find "$D" -maxdepth 1 -type f \
      \( -iname 'LDEC_80S_20MPP_ADJ.*' \
         -o -iname 'LDEC_83S_10MPP_ADJ.*' \
         -o -iname 'ldec_87s_5mpp.tif' \
         -o -iname '*_final_adj_5mpp_ldec.tif' \) \
      -print0 | sort -z
)

ELEV_UNC_ARGS=()
while IFS= read -r -d '' f; do
    ELEV_UNC_ARGS+=(--elev-uncertainty "$f")
    echo "ELEVATION UNCERTAINTY: $(basename "$f")"
done < <(
    find "$D" -maxdepth 1 -type f \
      -iname '*_final_adj_5mpp_toterr.tif' -print0
)

SLOPE_UNC_ARGS=()
while IFS= read -r -d '' f; do
    SLOPE_UNC_ARGS+=(--slope-uncertainty "$f")
    echo "SLOPE UNCERTAINTY: $(basename "$f")"
done < <(
    find "$D" -maxdepth 1 -type f \
      -iname '*_final_adj_5mpp_slperr.tif' -print0
)

SLOPE_CHECK_ARGS=()
while IFS= read -r -d '' f; do
    SLOPE_CHECK_ARGS+=(--slope-check "$f")
    echo "SLOPE CHECK: $(basename "$f")"
done < <(
    find "$D" -maxdepth 1 -type f \
      \( -iname 'ldsm_87s_5mpp.tif' \
         -o -iname '*_final_adj_5mpp_slp.tif' \) \
      -print0
)

OUT="derived/model3_${TAG}"
PACK="data/terrain_${TAG}"

mkdir -p "$OUT" "$PACK"

python tools/datakit.py dem \
    "${DEM_ARGS[@]}" \
    --sites "$SITES" \
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
    --out "$OUT" \
    --allow-large

cp -v "$OUT"/terrain/* "$PACK"/

python tools/build_model3_comparison.py \
    --sites "$SITES" \
    --terrain-dir "$PACK" \
    --output "data/model3_${TAG}_comparison.csv" \
    --slope-map-output "static/model3_${TAG}_slope_map.png"

echo
echo "======================================"
echo "TERRAIN RUN FINISHED"
echo "Processed sites: $SITES"
echo "Terrain products: $PACK"
echo "Comparison: data/model3_${TAG}_comparison.csv"
echo "Slope map: static/model3_${TAG}_slope_map.png"
echo "======================================"
