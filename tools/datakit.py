#!/usr/bin/env python3
"""
datakit - take only what we need out of multi-GB datasets.

    python tools/datakit.py auto    FILE [--sites data/site_comparison.csv]   # identify any file, suggest the command
    python tools/datakit.py inspect FILE [--sites ...]                        # describe a file WITHOUT loading it
    python tools/datakit.py dem     --dem near.jp2:10000 --dem far.jp2:150000 --sites SITES.csv   # LOLA / any DEM
    python tools/datakit.py spice   --kernels KERNELDIR --start 2026-10-15T00:00:00 --days 89 --sites SITES.csv
    python tools/datakit.py table   FILE --columns a,b --where "lat < -80" --every 10            # any big CSV/Parquet

Design rules
  * Huge inputs are only ever READ in windows / chunks, never loaded whole.
  * Outputs are small, committable "derived products" (default folder: derived/).
  * Every output gets a  <file>.provenance.json  (source, fingerprint, parameters, command, date) and an
    entry in derived/MANIFEST.json, so anybody can see where a number came from.
  * Outputs bigger than 50 MB are refused unless --allow-large (GitHub limit is 100 MB per file).
  * Nothing is guessed silently: unknown projection or unit scaling stops the run or prints a loud warning.

Sites file: any CSV with columns like site_id, latitude_deg, longitude_deg (names are matched loosely;
optional site_name and observer_elevation_m).  data/site_comparison.csv works as is.
"""
import argparse
import contextlib
import gzip
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "0.1"
MOON_RADIUS_M = 1_737_400.0
EARTH_RADIUS_KM = 6378.137

RASTER_EXT = {".tif", ".tiff", ".jp2", ".img", ".lbl", ".vrt", ".asc", ".nc"}
SPICE_EXT = {".bsp", ".bpc", ".bc", ".tf", ".tpc", ".tls", ".tsc", ".ti", ".tm", ".bds"}
TABLE_EXT = {".csv", ".tsv", ".txt", ".parquet", ".pq"}


# ============================================================ small utilities
def is_url(s) -> bool:
    return bool(re.match(r"^https?://", str(s)))


def say(msg=""):
    print(msg, flush=True)


def fingerprint(path) -> dict:
    """Cheap identity of a huge file: size + SHA-256 of its first and last megabyte."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if size > (2 << 20):
            f.seek(-(1 << 20), 2)
            h.update(f.read())
    return {"path": str(p), "size_bytes": size, "sha256_first_last_MB": h.hexdigest()}


def source_info(path) -> dict:
    return {"url": str(path)} if is_url(path) else fingerprint(path)


def write_provenance(out_path, source, parameters, extra=None):
    meta = {"tool": f"datakit {VERSION}", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv), "source": source, "parameters": parameters, **(extra or {})}
    Path(str(out_path) + ".provenance.json").write_text(json.dumps(meta, indent=2, default=str))


def register_manifest(outdir, files, source, tool_step):
    path = Path(outdir) / "MANIFEST.json"
    entries = json.loads(path.read_text()) if path.exists() else []
    for f in files:
        entries.append({"file": str(f), "step": tool_step, "source": source if isinstance(source, str) else str(source),
                        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "size_bytes": Path(f).stat().st_size})
    path.write_text(json.dumps(entries, indent=2))


def guard_size(path, allow_large=False):
    mb = Path(path).stat().st_size / 1e6
    if mb > 50 and not allow_large:
        Path(path).unlink()
        raise SystemExit(f"Refusing to keep {path} ({mb:.0f} MB > 50 MB). Use a smaller --radius / fewer columns, "
                         "or pass --allow-large if you really mean it. The file was deleted.")
    if mb > 10:
        say(f"  note: {path} is {mb:.0f} MB - fine for git, but consider whether it is all needed.")


def read_sites(path, only=None):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names, required=True):
        for n in names:
            if n in cols:
                return cols[n]
        if required:
            raise SystemExit(f"{path}: need a column like one of {names}; found {list(df.columns)}")
        return None

    id_c = pick("site_id", "id", "site")
    lat_c = pick("latitude_deg", "lat", "latitude", "lat_deg")
    lon_c = pick("longitude_deg", "lon", "longitude", "lon_deg")
    name_c = pick("site_name", "name", required=False)
    el_c = pick("observer_elevation_m", "elevation_m", "site_elevation_m", "dem_elevation_at_site_m", required=False)
    sites = [{"id": str(r[id_c]), "name": str(r[name_c]) if name_c else str(r[id_c]), "lat": float(r[lat_c]),
              "lon": float(r[lon_c]), "elev_m": float(r[el_c]) if el_c and pd.notna(r[el_c]) else None}
             for _, r in df.iterrows()]
    if only:
        sites = [s for s in sites if s["id"] in set(only)]
        if not sites:
            raise SystemExit(f"None of {only} found in {path}")
    return sites


def destination(lat_deg, lon_deg, az_deg, dist_m, radius_m=MOON_RADIUS_M):
    """Great-circle destination on a sphere (vectorised). Azimuth is clockwise from north."""
    lat1, lon1, th, d = np.radians(lat_deg), np.radians(lon_deg), np.radians(az_deg), np.asarray(dist_m) / radius_m
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(th))
    lon2 = lon1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lon2), np.degrees(lat2)


def sniff_sep(path) -> str:
    name = str(path).lower()
    if name.endswith((".tsv", ".tsv.gz")):
        return "\t"
    if is_url(path):
        return ","
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt", errors="ignore") as f:
        line = f.readline()
    return max([",", "\t", ";", "|"], key=line.count)


def detect_kind(path) -> str:
    name = str(path).lower()
    name = re.sub(r"\.gz$", "", name)
    ext = Path(name).suffix
    if ext in RASTER_EXT:
        return "raster"
    if ext in SPICE_EXT:
        return "spice"
    if ext in TABLE_EXT:
        return "table"
    if not is_url(path) and Path(path).exists() and Path(path).is_file():
        head = Path(path).open("rb").read(8)
        if head[:4] in (b"II*\x00", b"MM\x00*") or head[:4] == b"\x00\x00\x00\x0c":
            return "raster"
        if head.startswith(b"DAF/") or head.startswith(b"NAIF"):
            return "spice"
    return "unknown"


# ============================================================ raster helpers (LOLA / any DEM)
@contextlib.contextmanager
def raster(path):
    import rasterio
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
               CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".JP2,.IMG,.tif,.TIF,.tiff,.LBL,.lbl")
    with rasterio.Env(**env):
        try:
            src = rasterio.open(path)
        except Exception:
            alt = None
            if not is_url(path) and str(path).lower().endswith(".img"):
                for ext in (".LBL", ".lbl"):
                    cand = Path(path).with_suffix(ext)
                    if cand.exists():
                        alt = cand
            if alt is None:
                raise
            say(f"  (opening the label {alt.name} instead of the .IMG)")
            src = rasterio.open(alt)
        try:
            yield src
        finally:
            src.close()


def pds_label_scaling(path):
    """Find SCALING_FACTOR / OFFSET in a PDS3 .LBL or PDS4 .xml sitting next to a local file."""
    if is_url(path):
        return None
    p = Path(path)
    stem = p.stem.split(".")[0]
    for f in sorted(p.parent.iterdir()):
        if f.suffix.lower() in (".lbl", ".xml") and f.stem.lower().startswith(stem.lower()):
            text = f.read_text(errors="ignore")
            s = re.search(r"^\s*SCALING_FACTOR\s*=\s*([-+0-9.eEdD]+)", text, re.M) or re.search(r"<scaling_factor[^>]*>\s*([-+0-9.eE]+)", text)
            o = re.search(r"^\s*OFFSET\s*=\s*([-+0-9.eEdD]+)", text, re.M) or re.search(r"<value_offset[^>]*>\s*([-+0-9.eE]+)", text)
            if s or o:
                num = lambda m: float(m.group(1).replace("D", "E").replace("d", "e")) if m else None
                return {"file": f.name, "scale": num(s), "offset": num(o)}
    return None


def determine_scaling(src, path, scale_arg, offset_arg):
    """CLI override > GDAL metadata > PDS label > assume 1/0 (loudly)."""
    gdal_s = src.scales[0] if src.scales else 1.0
    gdal_o = src.offsets[0] if src.offsets else 0.0
    label = pds_label_scaling(path)
    if scale_arg is not None or offset_arg is not None:
        scale, offset, how = (scale_arg if scale_arg is not None else 1.0), (offset_arg if offset_arg is not None else 0.0), "command line"
    elif (gdal_s, gdal_o) != (1.0, 0.0):
        scale, offset, how = gdal_s, gdal_o, "GDAL metadata"
    elif label and (label["scale"] is not None or label["offset"] is not None):
        scale, offset, how = label["scale"] or 1.0, label["offset"] or 0.0, f"PDS label {label['file']}"
    else:
        scale, offset, how = 1.0, 0.0, "ASSUMED (no scale/offset found in the file or a label)"
    say(f"  scaling: scale={scale:g} offset={offset:g}  [{how}]")
    if how.startswith("ASSUMED"):
        say("  WARNING: confirm the unit scaling in the dataset's label/documentation (LOLA integers are often scaled).")
    if label and how == "GDAL metadata" and (label["scale"], label["offset"]) != (gdal_s, gdal_o):
        say(f"  WARNING: GDAL says {gdal_s}/{gdal_o} but label {label['file']} says {label['scale']}/{label['offset']}.")
    return scale, offset, how


def to_heights(raw, scale, offset):
    """Raw masked integers -> float32 metres above the reference sphere (NaN where missing)."""
    eff = offset - MOON_RADIUS_M if offset > 1e6 else offset            # fold the 1,737,400 m radius into the offset
    z = raw.astype("float32") * np.float32(scale) + np.float32(eff)
    if np.ma.median(z) > 1e6:
        z = z - np.float32(MOON_RADIUS_M)
    z = np.ma.filled(z, np.nan).astype("float32")
    med = float(np.nanmedian(z)) if np.isfinite(z).any() else float("nan")
    if not np.isfinite(med) or abs(med) > 20000:
        say(f"  WARNING: median height {med:.0f} m is implausible for the Moon - check scale/offset/datum.")
    return z


def geo_to_proj(crs):
    from pyproj import CRS, Transformer
    geo = CRS.from_proj4(f"+proj=longlat +R={MOON_RADIUS_M} +type=crs")
    return Transformer.from_crs(geo, crs, always_xy=True)


def require_crs(src, assume_crs=None):
    if src.crs is not None:
        crs = src.crs
    elif assume_crs:
        from pyproj import CRS
        say(f"  WARNING: file has no CRS; ASSUMING {assume_crs}")
        crs = CRS.from_proj4(assume_crs)
    else:
        raise SystemExit("The raster has no CRS. Read its label/documentation and pass --assume-crs '<proj4>' explicitly; "
                         "refusing to guess a projection.")
    if getattr(crs, "is_geographic", False):
        raise SystemExit("A geographic (degree) CRS cannot be used for metre-based terrain radii/slopes. "
                         "Reproject the DEM to a lunar polar metric CRS first.")
    units = getattr(crs, "linear_units_factor", None)
    if units and abs(float(units[1]) - 1.0) > 1e-9:
        raise SystemExit(f"DEM CRS linear units are {units[0]!r}, not metres. Reproject to metres before processing.")
    return crs


def window_around(src, x, y, half_m):
    from rasterio.windows import Window
    row0, col0 = src.index(x, y)
    hw, hh = int(round(half_m / abs(src.res[0]))), int(round(half_m / abs(src.res[1])))
    c0, r0 = max(0, col0 - hw), max(0, row0 - hh)
    c1, r1 = min(src.width, col0 + hw + 1), min(src.height, row0 + hh + 1)
    return Window(c0, r0, c1 - c0, r1 - r0), (row0 - r0, col0 - c0), ((r1 - r0) != 2 * hh + 1 or (c1 - c0) != 2 * hw + 1)


# ------------------------------------------------------------ products
def horizon_for_site(bands, site, observer_height, az_step, assume_crs=None):
    """Terrain horizon elevation per azimuth, max over distance bands (near/mid/far DEMs)."""
    azs = np.arange(0, 360, az_step, dtype=float)
    per_band, found, z0 = [], [], None
    for b in bands:
        with raster(b["path"]) as src:
            crs = require_crs(src, assume_crs)
            tf = geo_to_proj(crs)
            x, y = tf.transform(site["lon"], site["lat"])
            if not (src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top):
                raise SystemExit(f"{site['id']} is outside {Path(b['path']).name}")
            step = max(abs(src.res[0]), abs(src.res[1]))
            win, (r_site, c_site), _ = window_around(src, x, y, b["max_d"] + 2 * step)
            h = to_heights(src.read(1, window=win, masked=True), b["scale"], b["offset"])
            wtr = src.window_transform(win)
        if z0 is None:
            z0 = float(h[r_site, c_site])
            if not np.isfinite(z0):
                raise SystemExit(f"{site['id']}: elevation missing at the site pixel of {Path(b['path']).name}")
        d = np.arange(step, b["max_d"] + step / 2, step)
        lon2, lat2 = destination(site["lat"], site["lon"], azs[:, None], d[None, :])
        xs, ys = tf.transform(lon2.ravel(), lat2.ravel())
        inv = ~wtr
        col = np.floor(inv.a * xs + inv.b * ys + inv.c).astype(int)
        row = np.floor(inv.d * xs + inv.e * ys + inv.f).astype(int)
        ok = (row >= 0) & (row < h.shape[0]) & (col >= 0) & (col < h.shape[1])
        vals = np.full(xs.shape, np.nan)
        vals[ok] = h[row[ok], col[ok]]
        vals = vals.reshape(len(azs), len(d))
        r_t = MOON_RADIUS_M + vals
        central = d[None, :] / MOON_RADIUS_M
        ang = np.degrees(np.arctan2(r_t * np.cos(central) - (MOON_RADIUS_M + z0 + observer_height), r_t * np.sin(central)))
        valid = np.isfinite(ang)
        hz = np.where(valid, ang, -np.inf).max(axis=1)
        hz[~valid.any(axis=1)] = np.nan
        per_band.append(hz)
        found.append(valid.sum(axis=1))
    stack = np.vstack(per_band)
    final = np.where(np.isfinite(stack), stack, -np.inf).max(axis=0)
    final[~np.isfinite(stack).any(axis=0)] = np.nan
    return azs, final, per_band, np.sum(found, axis=0), z0


def _masked_derivative(a, spacing, axis):
    """Finite difference on valid neighbours only; never bridge nodata gaps."""
    z = np.asarray(a, dtype=np.float64)
    out = np.full(z.shape, np.nan, dtype=np.float64)
    if z.shape[axis] < 2:
        return out
    mid = [slice(None)] * z.ndim
    prev = [slice(None)] * z.ndim
    nxt = [slice(None)] * z.ndim
    mid[axis], prev[axis], nxt[axis] = slice(1, -1), slice(0, -2), slice(2, None)
    m, p, n = tuple(mid), tuple(prev), tuple(nxt)
    vm, vp, vn = np.isfinite(z[m]), np.isfinite(z[p]), np.isfinite(z[n])
    both = vm & vp & vn
    tmp = np.full(z[m].shape, np.nan, dtype=np.float64)
    tmp[both] = (z[n][both] - z[p][both]) / (2.0 * spacing)
    only_p = vm & vp & ~vn
    tmp[only_p] = (z[m][only_p] - z[p][only_p]) / spacing
    only_n = vm & vn & ~vp
    tmp[only_n] = (z[n][only_n] - z[m][only_n]) / spacing
    out[m] = tmp
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


def _block_mean_nan(a, factor):
    """NaN-aware block mean, padding rather than discarding edge pixels."""
    a = np.asarray(a, dtype=np.float64)
    h, w = a.shape
    hp = int(math.ceil(h / factor) * factor)
    wp = int(math.ceil(w / factor) * factor)
    padded = np.full((hp, wp), np.nan, dtype=np.float64)
    padded[:h, :w] = a
    blocks = padded.reshape(hp // factor, factor, wp // factor, factor)
    valid = np.isfinite(blocks)
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
    count = valid.sum(axis=(1, 3))
    out = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def align_auxiliary_layer(layer, target_src, target_window, target_crs, shape):
    """Windowed GDAL warp of a companion raster onto the DEM cutout grid."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    out = np.full(shape, np.nan, dtype="float32")
    with raster(layer["path"]) as aux:
        aux_crs = require_crs(aux, layer.get("assume_crs"))
        reproject(
            source=rasterio.band(aux, 1),
            destination=out,
            src_transform=aux.transform,
            src_crs=aux_crs,
            src_nodata=aux.nodata,
            dst_transform=target_src.window_transform(target_window),
            dst_crs=target_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )
    out = out * np.float32(layer.get("scale", 1.0)) + np.float32(layer.get("offset", 0.0))
    if layer.get("units") == "percent":
        out = np.degrees(np.arctan(out / 100.0)).astype("float32")
    return out


def terrain_products(src, tf, site, radius, max_grid, scale, offset, outdir, source, params, allow_large, auxiliary_layers=None, assume_crs=None, footprint_radius_m=None):
    x, y = tf.transform(site["lon"], site["lat"])
    win, (rs, cs), clipped = window_around(src, x, y, radius)
    if clipped:
        raise SystemExit(f"{site['id']}: radius {radius} m runs past {Path(source).name} edge.")
    px, py = abs(src.res[0]), abs(src.res[1])
    z = to_heights(src.read(1, window=win, masked=True), scale, offset).astype("float64")
    if not (0 <= rs < z.shape[0] and 0 <= cs < z.shape[1]):
        raise SystemExit(f"{site['id']}: site pixel is outside the extracted DEM window.")
    site_z = float(z[rs, cs])
    if not np.isfinite(site_z):
        raise SystemExit(f"{site['id']}: site elevation is nodata in {Path(source).name}.")
    dzdy = _masked_derivative(z, py, axis=0)
    dzdx = _masked_derivative(z, px, axis=1)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    slope[~np.isfinite(z)] = np.nan

    dirs = {}
    for label, az in (("north_xy", 0.0), ("east_xy", 90.0)):
        lo, la = destination(site["lat"], site["lon"], az, 100.0)
        x2, y2 = tf.transform(float(lo), float(la))
        v = np.array([x2 - x, y2 - y])
        dirs[label] = [round(float(c), 8) for c in v / np.linalg.norm(v)]

    # Aspect is clockwise from true north, not merely from the raster's y-axis.
    # Convert projected-grid downslope vectors into local east/north coordinates.
    basis = np.column_stack((np.asarray(dirs["east_xy"]), np.asarray(dirs["north_xy"])))
    if abs(np.linalg.det(basis)) < 1e-8:
        raise SystemExit(f"{site['id']}: projected grid axes are degenerate; cannot derive aspect.")
    down_map = np.vstack((-dzdx.ravel(), dzdy.ravel()))
    down_enu = np.linalg.solve(basis, down_map)
    aspect = (np.degrees(np.arctan2(down_enu[0], down_enu[1])) + 360.0) % 360.0
    aspect = aspect.reshape(z.shape)
    aspect[~np.isfinite(slope) | (slope < 0.1)] = np.nan

    # Local elevation standard deviation at an approximately 25 m physical scale.
    # Coarser DEMs necessarily have a larger minimum resolvable window.
    rough_window_px = max(3, int(round(25.0 / math.sqrt(px * py))))
    if rough_window_px % 2 == 0:
        rough_window_px += 1
    try:
        from scipy.ndimage import uniform_filter
        zvalid = np.isfinite(z)
        weights = uniform_filter(zvalid.astype("float64"), size=rough_window_px, mode="constant")
        sums = uniform_filter(np.where(zvalid, z, 0.0), size=rough_window_px, mode="constant")
        sums2 = uniform_filter(np.where(zvalid, z * z, 0.0), size=rough_window_px, mode="constant")
        mean = np.divide(sums, weights, out=np.full_like(sums, np.nan), where=weights >= 0.8)
        mean2 = np.divide(sums2, weights, out=np.full_like(sums2, np.nan), where=weights >= 0.8)
        roughness = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
        roughness[~zvalid] = np.nan
    except ImportError:
        roughness = np.full(z.shape, np.nan, dtype="float64")

    f = max(1, int(math.ceil(max(z.shape) / max_grid)))
    height = (_block_mean_nan(z, f) - site_z).astype("float32")
    slope_c = _block_mean_nan(slope, f).astype("float32")
    roughness_c = _block_mean_nan(roughness, f).astype("float32")
    aspect_sin = _block_mean_nan(np.sin(np.radians(aspect)), f)
    aspect_cos = _block_mean_nan(np.cos(np.radians(aspect)), f)
    aspect_c = (np.degrees(np.arctan2(aspect_sin, aspect_cos)) + 360.0) % 360.0
    aspect_c[~np.isfinite(aspect_sin) | ~np.isfinite(aspect_cos)] = np.nan
    aspect_c = aspect_c.astype("float32")
    valid_mask = np.isfinite(height)

    aligned_aux = {}
    target_crs = require_crs(src, assume_crs)
    for key, layer in (auxiliary_layers or {}).items():
        if layer is not None:
            aligned = align_auxiliary_layer(layer, src, win, target_crs, z.shape)
            aligned_aux[key] = _block_mean_nan(aligned, f).astype("float32")

    site_col = (cs - (f - 1) / 2.0) / f
    site_row = (rs - (f - 1) / 2.0) / f
    source_mpp = math.sqrt(px * py)
    grid_mpp = source_mpp * f
    n_valid = int(valid_mask.sum())
    n_total = int(valid_mask.size)
    meta = {
        "name": site["id"].lower(), "site_id": site["id"],
        "lat": site["lat"], "lon": site["lon"],
        "grid_px": [int(height.shape[1]), int(height.shape[0])],
        "site_pixel": [float(site_col), float(site_row)],
        "metres_per_pixel": grid_mpp,
        "metres_per_pixel_xy": [px * f, py * f],
        "source_metres_per_pixel": source_mpp,
        "site_elevation_m": site_z,
        "aspect_reference": "clockwise from local true north; downslope direction",
        "roughness_definition": "local elevation standard deviation",
        "roughness_window_pixels": rough_window_px,
        "roughness_window_approx_m": rough_window_px * source_mpp,
        "grid_half_width_m": float(radius),
        "footprint_radius_m": float(footprint_radius_m if footprint_radius_m is not None else radius),
        "valid_grid_pixels": n_valid,
        "grid_coverage_pct": round(100.0 * n_valid / n_total, 5) if n_total else 0.0,
        "valid_grid_area_km2": n_valid * px * py * f * f / 1e6,
        "relief_min_m": float(np.nanmin(height)) if np.isfinite(height).any() else None,
        "relief_max_m": float(np.nanmax(height)) if np.isfinite(height).any() else None,
        "grid_slope_stats_native_resolution": {
            f"area_below_{t}deg_pct": round(float(100 * np.mean(slope[np.isfinite(slope)] < t)), 2)
            for t in (5, 10, 15, 20)
        } if np.isfinite(slope).any() else {},
        **dirs,
        "dem_file": Path(str(source)).name,
        "source_crs": require_crs(src, assume_crs).to_string(),
        "source_transform": tuple(src.transform),
        "source_nodata": src.nodata,
        "elevation_scale": float(scale), "elevation_offset": float(offset),
        "scaling_method": params.get("scale_how", "not recorded"),
        "auxiliary_layers": {
            key: {"file": Path(value["path"]).name, "scale": value.get("scale", 1.0),
                  "offset": value.get("offset", 0.0), "units": value.get("units", "not specified")}
            for key, value in (auxiliary_layers or {}).items() if value is not None
        },
        "grid_convention": "x grows with column, y decreases with row; site_pixel locates the site",
        "note": "height_m is relative to the site; slope derived at native DEM resolution before block averaging; nodata retained."
    }
    out = Path(outdir) / "terrain"
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{site['id'].lower()}_terrain"
    arrays = {
        "height_m": height,
        "slope_deg": slope_c,
        "aspect_deg": aspect_c,
        "roughness_m": roughness_c,
        "valid_mask": valid_mask.astype("uint8"),
    }
    arrays.update(aligned_aux)
    # A supplied slope is retained as an independent cross-check; derived slope
    # remains the Model 3 calculation used for screening.
    if "slope_check_deg" in aligned_aux:
        common = np.isfinite(slope_c) & np.isfinite(aligned_aux["slope_check_deg"])
        difference = np.full(slope_c.shape, np.nan, dtype="float32")
        difference[common] = aligned_aux["slope_check_deg"][common] - slope_c[common]
        arrays["slope_check_difference_deg"] = difference
    np.savez_compressed(str(base) + ".npz", **arrays)
    Path(str(base) + ".json").write_text(json.dumps(meta, indent=2))
    for ext in (".npz", ".json"):
        guard_size(str(base) + ext, allow_large)
        source_records = [source_info(source)] + [
            source_info(layer["path"]) for layer in (auxiliary_layers or {}).values() if layer is not None
        ]
        write_provenance(str(base) + ext, source_records, params, extra={"terrain_metadata": meta})
    return [str(base) + ".npz", str(base) + ".json"]


def select_best_terrain_dem(bands, site, radius, assume_crs=None, required=True, label="terrain raster"):
    """Choose the finest raster that contains the complete requested site footprint."""
    from rasterio.windows import Window
    candidates, rejected = [], []
    for band in bands:
        with raster(band["path"]) as src:
            try:
                crs = require_crs(src, assume_crs)
                tf = geo_to_proj(crs)
                x, y = tf.transform(site["lon"], site["lat"])
                win, (rs, cs), clipped = window_around(src, x, y, radius)
                if clipped:
                    rejected.append(f"{Path(band['path']).name}: does not cover full {radius:g} m radius")
                    continue
                site_window = Window(win.col_off + cs, win.row_off + rs, 1, 1)
                sample = src.read(1, window=site_window, masked=True)
                if sample.size == 0 or np.ma.is_masked(sample[0, 0]) or not np.isfinite(float(sample[0, 0])):
                    rejected.append(f"{Path(band['path']).name}: site pixel is nodata")
                    continue
                resolution = math.sqrt(abs(src.res[0] * src.res[1]))
                candidates.append((resolution, band["path"], band))
            except Exception as exc:
                rejected.append(f"{Path(band['path']).name}: {exc}")
    if not candidates:
        detail = "\n  ".join(rejected)
        if not required:
            say(f"  no {label} covers the complete footprint; continuing without it")
            return None
        raise SystemExit(f"{site['id']}: no DEM covers the full {radius:g} m terrain footprint.\n  {detail}")
    candidates.sort(key=lambda item: item[0])
    chosen = candidates[0][2]
    say(f"  selected {label}: {Path(chosen['path']).name} "
        f"({candidates[0][0]:g} m/pixel; full {radius:g} m footprint covered)")
    for res, path, _ in candidates[1:]:
        say(f"  available fallback: {Path(path).name} ({res:g} m/pixel)")
    return chosen

def cutout_product(src, tf, site, radius, scale, offset, outdir, source, name):
    import rasterio
    x, y = tf.transform(site["lon"], site["lat"])
    win, _, _ = window_around(src, x, y, radius)
    z = to_heights(src.read(1, window=win, masked=True), scale, offset)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{site['id'].lower()}_{name}.tif"
    prof = dict(driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, dtype="float32", crs=src.crs,
                transform=src.window_transform(win), nodata=np.nan, tiled=True, blockxsize=256, blockysize=256,
                compress="deflate", predictor=3, zlevel=9)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(z, 1)
        dst.update_tags(source=str(source), scale=str(scale), offset=str(offset), note="height above reference sphere, metres")
    write_provenance(path, source_info(source), {"radius_m": radius, "scale": scale, "offset": offset})
    return str(path)


# ============================================================ command: inspect / auto
def cmd_inspect(a):
    path = a.file
    kind = detect_kind(path)
    say(f"{path}\n  detected type: {kind}")
    if not is_url(path) and Path(path).exists():
        say(f"  size on disk: {Path(path).stat().st_size / 1e6:,.1f} MB")
    if kind == "raster":
        inspect_raster(path, a.sites)
    elif kind == "spice":
        inspect_spice(path, a)
    elif kind == "table":
        inspect_table(path)
    else:
        say("  Unknown format. Open it manually, then tell me what it is and I can add a reader.")
    return kind


def inspect_raster(path, sites_path):
    with raster(path) as src:
        say(f"  size: {src.width} x {src.height} px, {src.count} band(s), dtype {src.dtypes[0]}, resolution {src.res}")
        say(f"  uncompressed: about {src.width * src.height * np.dtype(src.dtypes[0]).itemsize / 1e9:,.2f} GB  ->  never load whole; use `dem`")
        say(f"  CRS: {(src.crs.to_string()[:110] if src.crs else 'NONE (must be supplied with --assume-crs)')}")
        say(f"  bounds: {tuple(round(v, 1) for v in src.bounds)}   nodata: {src.nodata}")
        say(f"  GDAL scales/offsets: {src.scales}/{src.offsets}")
        lab = pds_label_scaling(path)
        say(f"  PDS label scaling: {lab if lab else 'no label found next to the file'}")
        if sites_path and src.crs is not None:
            tf = geo_to_proj(src.crs)
            say("  site coverage (distance to the nearest edge = the largest radius this file supports):")
            for s in read_sites(sites_path):
                x, y = tf.transform(s["lon"], s["lat"])
                b = src.bounds
                if b.left <= x <= b.right and b.bottom <= y <= b.top:
                    edge = min(x - b.left, b.right - x, y - b.bottom, b.top - y)
                    say(f"    {s['id']:8s} inside   (max radius ~{edge / 1000:,.1f} km)")
                else:
                    say(f"    {s['id']:8s} OUTSIDE  - needs a different tile")


def inspect_spice(path, a):
    ext = Path(str(path).lower()).suffix
    try:
        import spiceypy as spice
    except ImportError:
        spice = None
    if ext in (".tf", ".tpc", ".tls", ".tm", ".ti", ".tsc"):
        text = Path(path).read_text(errors="ignore")
        say(f"  text kernel; first lines: {[l.strip() for l in text.splitlines() if l.strip()][:3]}")
        frames = re.findall(r"FRAME_([A-Z0-9_]+)\s*=\s*\d+", text)
        if frames:
            say(f"  frames defined: {sorted(set(frames))[:12]}")
        if ext == ".tls":
            leaps = re.findall(r"(\d+),\s*@(\d{4}-[A-Z]{3}-\d)", text)
            say(f"  last leap second entry: {leaps[-1] if leaps else 'n/a'}")
        if ext == ".tm":
            say(f"  kernels listed: {re.findall(r'[\'\"]([^\'\"]+)[\'\"]', text)[:10]}")
        return
    if spice is None:
        say("  (install spiceypy to see time coverage: pip install spiceypy)")
        return
    lsk = next((str(f) for f in Path(path).parent.glob("*.tls")), None)
    if lsk:
        spice.furnsh(lsk)
    try:
        if ext == ".bsp":
            ids = spice.spkobj(str(path))
            say(f"  SPK with {len(ids)} body id(s): {sorted(ids)[:15]}")
            for i in sorted(ids)[:8]:
                cov = spice.spkcov(str(path), i)
                for k in range(spice.wncard(cov)):
                    t0, t1 = spice.wnfetd(cov, k)
                    f = (lambda t: spice.et2utc(t, "ISOC", 0)) if lsk else (lambda t: f"ET {t:.0f}")
                    say(f"    body {i}: {f(t0)}  ->  {f(t1)}")
        elif ext == ".bpc":
            ids = spice.pckfrm(str(path))
            say(f"  binary PCK covering frame id(s): {sorted(ids)}")
        else:
            say("  binary SPICE kernel (use NAIF's `brief` tool for full detail)")
        if not lsk:
            say("  (put a leap-seconds .tls kernel in the same folder to see coverage in UTC)")
    finally:
        spice.kclear()


def inspect_table(path):
    if str(path).lower().endswith((".parquet", ".pq")):
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            say(f"  parquet: {pf.metadata.num_rows:,} rows in {pf.metadata.num_row_groups} row groups")
            say(f"  columns: {pf.schema_arrow.names}")
        except ImportError:
            say("  pip install pyarrow to inspect parquet files")
        return
    sep = sniff_sep(path)
    sample = pd.read_csv(path, nrows=2000, sep=sep)
    say(f"  delimiter {sep!r}; columns ({len(sample.columns)}): {list(sample.columns)}")
    say(f"  dtypes: {dict(sample.dtypes.astype(str))}")
    if not is_url(path) and not str(path).lower().endswith(".gz"):
        per_row = len(sample.to_csv(index=False, sep=sep).encode()) / len(sample)
        say(f"  rough row count: ~{int(Path(path).stat().st_size / per_row):,} (file size / average row; a guess)")
    say("  first rows:\n" + sample.head(3).to_string())


def cmd_auto(a):
    kind = cmd_inspect(a)
    say("\nSuggested next step:")
    if kind == "raster":
        say(f"  python tools/datakit.py dem --dem {a.file}:10000 --sites {a.sites or 'data/site_comparison.csv'} --products terrain,horizon")
        say("  (add more --dem FILE:MAXDIST bands, finest first, to build a horizon that sees far terrain)")
    elif kind == "spice":
        say("  python tools/datakit.py spice --kernels <folder with the kernels> --start 2026-10-15T00:00:00 --days 89 "
            "--sites data/site_comparison.csv")
    elif kind == "table":
        say(f"  python tools/datakit.py table {a.file} --columns <a,b,c> --where \"<expr>\" --every 10 --out derived/<name>.csv")


# ============================================================ command: dem
def parse_dem_spec(spec, default_max):
    """'path[:max_distance_m]'. Handles URLs (https://...) and Windows drive letters (C:\\...)."""
    m = re.match(r"^(.*):(\d+(?:\.\d+)?)$", spec)
    if m and not re.match(r"^[A-Za-z]$", m.group(1)) and m.group(1).lower() not in ("http", "https"):
        return {"path": m.group(1), "max_d": float(m.group(2))}
    return {"path": spec, "max_d": default_max}


def cmd_dem(a):
    products = {p.strip() for p in a.products.split(",")}
    unknown = products - {"cutout", "terrain", "horizon"}
    if unknown:
        raise SystemExit(f"Unknown product(s) {unknown}; choose from cutout, terrain, horizon")
    sites = read_sites(a.sites, a.site)
    if a.radius <= 0:
        raise SystemExit("--radius must be positive.")
    if a.footprint_radius is not None and (a.footprint_radius <= 0 or a.footprint_radius > a.radius):
        raise SystemExit("--footprint-radius must be positive and no larger than --radius (the grid half-width).")
    bands = [parse_dem_spec(s, a.radius) for s in a.dem]
    if any(b["max_d"] <= 0 for b in bands) or [b["max_d"] for b in bands] != sorted(b["max_d"] for b in bands):
        raise SystemExit("List --dem bands from finest/near to coarsest/far, with increasing MAX distances (e.g. a.jp2:10000 b.jp2:70000).")
    say(f"datakit dem: {len(sites)} site(s), {len(bands)} DEM band(s), products {sorted(products)}")
    for b in bands:
        say(f"\n[{Path(b['path']).name}] up to {b['max_d'] / 1000:g} km")
        with raster(b["path"]) as src:
            b["scale"], b["offset"], b["scale_how"] = determine_scaling(src, b["path"], a.scale, a.offset)
        if b["scale_how"].startswith("ASSUMED") and not a.allow_assumed_scaling:
            raise SystemExit(
                f"{Path(b['path']).name}: scaling is not documented in GDAL/PDS metadata. "
                "Verify the product units, then rerun with --allow-assumed-scaling only if raw pixels are already metres."
            )
    if a.slope_check and not a.slope_check_units:
        raise SystemExit("--slope-check requires --slope-check-units degrees or percent; units are never guessed.")
    aux_specs = {
        "count": (a.count, "count"),
        "elevation_uncertainty_m": (a.elev_uncertainty, "m"),
        "slope_uncertainty_deg": (a.slope_uncertainty, "degrees"),
        "slope_check_deg": (a.slope_check, a.slope_check_units),
    }
    aux_candidates = {}
    for key, (paths, units) in aux_specs.items():
        candidates = []
        for path in paths:
            item = {"path": path, "max_d": a.radius, "units": units, "assume_crs": a.assume_crs}
            with raster(path) as src:
                item["scale"], item["offset"], item["scale_how"] = determine_scaling(src, path, None, None)
            if item["scale_how"].startswith("ASSUMED") and not a.allow_assumed_scaling:
                raise SystemExit(
                    f"{Path(path).name}: companion-layer scaling is not documented. "
                    "Verify its units, then use --allow-assumed-scaling only if raw pixels are already in the declared units."
                )
            candidates.append(item)
        if candidates:
            aux_candidates[key] = candidates

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    produced = []
    source_desc = "; ".join(f"{Path(b['path']).name}<= {b['max_d']:g} m" for b in bands)

    if "horizon" in products:
        rows, diag = [], []
        for s in sites:
            say(f"\nhorizon: {s['id']} ...")
            azs, final, per_band, found, z0 = horizon_for_site(bands, s, a.observer_height, a.azimuth_step, a.assume_crs)
            if np.isnan(final).any():
                say(f"  WARNING: {int(np.isnan(final).sum())} azimuths have no valid terrain sample")
            gain = final - per_band[0]
            say(f"  site elevation {z0:.1f} m | horizon {np.nanmin(final):.2f}..{np.nanmax(final):.2f} deg | "
                f"farther bands raise the horizon by >0.1 deg in {int((gain > 0.1).sum())} of {len(azs)} directions "
                f"(max {np.nanmax(gain):.2f} deg)")
            for i, az in enumerate(azs):
                rows.append({"site_id": s["id"], "azimuth_deg": int(az) if float(az).is_integer() else az,
                             "horizon_elevation_deg": round(float(final[i]), 6), "terrain_samples_found": int(found[i])})
                diag.append({"site_id": s["id"], "azimuth_deg": az, **{f"band{k + 1}_deg": round(float(per_band[k][i]), 6) for k in range(len(bands))},
                             "final_deg": round(float(final[i]), 6), "far_field_gain_deg": round(float(gain[i]), 6)})
        hp, hd = outdir / "horizon_profiles.csv", outdir / "horizon_by_band.csv"
        pd.DataFrame(rows).to_csv(hp, index=False)
        pd.DataFrame(diag).to_csv(hd, index=False)
        params = {"observer_height_m": a.observer_height, "azimuth_step_deg": a.azimuth_step, "bands": [{k: b[k] for k in ("path", "max_d", "scale", "offset", "scale_how")} for b in bands],
                  "sites": [s["id"] for s in sites], "sphere_radius_m": MOON_RADIUS_M}
        for f in (hp, hd):
            guard_size(f, a.allow_large)
            write_provenance(f, [source_info(b["path"]) for b in bands], params)
            produced.append(str(f))
        say(f"\nwrote {hp} (same layout as the team's horizon_profiles.csv) and {hd} (per-band comparison)")

    if products & {"terrain", "cutout"}:
        for s in sites:
            say(f"\n{', '.join(sorted(products & {'terrain', 'cutout'}))}: {s['id']} ...")
            selected = select_best_terrain_dem(bands, s, a.radius, a.assume_crs)
            auxiliary = {}
            for key, candidates in aux_candidates.items():
                picked = select_best_terrain_dem(
                    candidates, s, a.radius, a.assume_crs,
                    required=False, label=f"{key} layer"
                )
                if picked is not None:
                    auxiliary[key] = picked
            with raster(selected["path"]) as src:
                crs = require_crs(src, a.assume_crs)
                tf = geo_to_proj(crs)
                params = {
                    "radius_m": a.radius, "max_grid": a.max_grid,
                    "scale": selected["scale"], "offset": selected["offset"],
                    "scale_how": selected["scale_how"],
                    "selection_rule": "finest DEM that covers the complete requested footprint",
                    "candidate_dem_files": [Path(b["path"]).name for b in bands],
                    "auxiliary_layers": {k: Path(v["path"]).name for k, v in auxiliary.items()},
                    "assume_crs": a.assume_crs,
                }
                if "terrain" in products:
                    produced += terrain_products(
                        src, tf, s, a.radius, a.max_grid,
                        selected["scale"], selected["offset"], outdir,
                        selected["path"], params, a.allow_large,
                        auxiliary_layers=auxiliary, assume_crs=a.assume_crs,
                        footprint_radius_m=a.footprint_radius
                    )
                if "cutout" in products:
                    produced.append(cutout_product(
                        src, tf, s, a.radius, selected["scale"], selected["offset"],
                        a.cutout_dir, selected["path"], "cutout"
                    ))
    register_manifest(outdir, produced, source_desc, "dem")
    say("\nfiles written:\n  " + "\n  ".join(produced))
    if "horizon" in products:
        say("\nNEXT: copy horizon_profiles.csv into data/ and run the validation suite.")
    if "terrain" in products:
        say("\nNEXT: copy derived terrain/*.npz and *.json into data/terrain/, then run the validation suite.")
    say("Keep raw multi-GB DEMs outside the app repository; retain each derived product's provenance files.")


# ============================================================ command: spice
SPICE_STD = [".tls", ".tpc", ".tf", ".bpc", ".bsp"]


def local_frame(lat_deg, lon_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    return up, east, north


def local_angles(vec_km, lat_deg, lon_deg, elev_m=0.0, radius_km=MOON_RADIUS_M / 1000):
    """Direction/distance of a body as seen FROM THE SITE ON THE SURFACE (not from the Moon's centre)."""
    up, east, north = local_frame(lat_deg, lon_deg)
    topo = np.asarray(vec_km, float) - (radius_km + elev_m / 1000.0) * up
    e, n, u = topo @ east, topo @ north, topo @ up
    return (np.degrees(np.arctan2(u, np.hypot(e, n))), np.degrees(np.arctan2(e, n)) % 360.0, np.linalg.norm(topo, axis=1))


def cmd_spice(a):
    import spiceypy as spice
    files = []
    for k in a.kernels:
        p = Path(k)
        if p.is_dir():
            files += sorted([f for f in p.iterdir() if f.suffix.lower() in SPICE_STD], key=lambda f: (SPICE_STD.index(f.suffix.lower()), f.name))
        else:
            files.append(p)
    if not files:
        raise SystemExit("No kernels found.")
    say("kernels loaded (in this order):")
    for f in files:
        say(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    if sum(f.suffix.lower() == ".bsp" for f in files) > 1 or sum(f.suffix.lower() == ".bpc" for f in files) > 1:
        say("  WARNING: several .bsp/.bpc files are loaded; later ones take precedence where they overlap. "
            "Pass only the kernels you need.")
    spice.kclear()
    try:
        for f in files:
            spice.furnsh(str(f))
        start = a.start.replace("T", " ")
        et0 = spice.str2et(start + " UTC")
        days = a.days if a.days is not None else (spice.str2et(a.end.replace("T", " ") + " UTC") - et0) / 86400.0
        n = int(round(days * 1440 / a.step_min))
        step_s = a.step_min * 60.0
        ets = et0 + np.arange(n) * step_s
        say(f"\n{n} epochs from {start} UTC, every {a.step_min:g} min ({days:g} days); frame {a.frame}, observer {a.observer}, abcorr {a.abcorr}")
        pos = {}
        for body in a.bodies:
            arr = np.empty((n, 3))
            for i, et in enumerate(ets):
                arr[i], _ = spice.spkpos(body, float(et), a.frame, a.abcorr, a.observer)
            pos[body] = arr
        utc = [spice.et2utc(float(et), "ISOC", 0) for et in ets]
    finally:
        spice.kclear()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.start[:10]}_{days:g}d_{a.step_min:g}m"
    npz = outdir / f"ephemeris_{tag}.npz"
    np.savez_compressed(npz, utc=np.array(utc), et=ets, **{f"{b.lower()}_km": v for b, v in pos.items()})
    guard_size(npz, a.allow_large)
    params = {"start_utc": a.start, "days": days, "step_min": a.step_min, "bodies": a.bodies, "observer": a.observer,
              "frame": a.frame, "abcorr": a.abcorr, "kernels": [f.name for f in files]}
    write_provenance(npz, [fingerprint(f) for f in files], params)
    produced = [str(npz)]
    say(f"wrote {npz} ({npz.stat().st_size / 1e6:.2f} MB): positions from the Moon's centre in {a.frame}, km")

    if a.sites:
        gdir = outdir / "local_geometry"
        gdir.mkdir(exist_ok=True)
        for s in read_sites(a.sites, a.site):
            cols = {"time_utc": utc}
            for b in a.bodies:
                el, az, dist = local_angles(pos[b], s["lat"], s["lon"], s["elev_m"] or 0.0)
                key = b.lower()
                cols.update({f"{key}_azimuth_deg": np.round(az, 6), f"{key}_elevation_deg": np.round(el, 6)})
                if key == "earth":
                    cols["earth_distance_km"] = np.round(dist, 3)
                    cols["earth_angular_radius_deg"] = np.round(np.degrees(np.arcsin(np.minimum(1, EARTH_RADIUS_KM / dist))), 6)
            path = gdir / f"{s['id']}_geometry.csv"
            pd.DataFrame(cols).to_csv(path, index=False)
            guard_size(path, a.allow_large)
            write_provenance(path, [fingerprint(f) for f in files], {**params, "site": s})
            produced.append(str(path))
            say(f"  {path} (angles seen from the surface site at {s['elev_m'] or 0:.0f} m elevation)")
    register_manifest(outdir, produced, ", ".join(f.name for f in files), "spice")
    say("\nThe app/validators can now use these tables with no SPICE kernels at all. Commit them; do NOT commit the kernels.")


# ============================================================ command: table
def cmd_table(a):
    path = a.file
    cols_wanted = [c.strip() for c in a.columns.split(",")] if a.columns else None
    parquet = str(path).lower().endswith((".parquet", ".pq"))
    sep = ","
    if parquet:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise SystemExit("pip install pyarrow to read parquet")
        header = pq.ParquetFile(path).schema_arrow.names
    else:
        sep = sniff_sep(path)
        header = list(pd.read_csv(path, nrows=0, sep=sep).columns)
    if cols_wanted:
        bad = [c for c in cols_wanted if c not in header]
        if bad:
            raise SystemExit(f"Unknown column(s) {bad}. Available: {header}")
    needed = set(cols_wanted or header)
    for extra in (a.where or "", a.time_col or "", a.lat_col if a.bbox else "", a.lon_col if a.bbox else ""):
        needed |= {t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", extra) if t in header}
    needed = [c for c in header if c in needed]
    bbox = a.bbox

    def chunks():
        if parquet:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=a.chunk, columns=needed):
                yield batch.to_pandas()
        else:
            yield from pd.read_csv(path, usecols=needed, chunksize=a.chunk, sep=sep)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scanned = kept = 0
    first = True
    for ch in chunks():
        scanned += len(ch)
        if a.time_col and (a.start or a.end):
            t = pd.to_datetime(ch[a.time_col], errors="coerce", utc=True)
            if a.start:
                ch = ch[t >= pd.Timestamp(a.start, tz="UTC")]
                t = t[ch.index]
            if a.end:
                ch = ch[t <= pd.Timestamp(a.end, tz="UTC")]
        if bbox:
            la, lo = ch[a.lat_col], ch[a.lon_col]
            ch = ch[(la >= bbox[0]) & (la <= bbox[1]) & (lo >= bbox[2]) & (lo <= bbox[3])]
        if a.where:
            ch = ch.query(a.where, engine="python")
        if a.every > 1:
            ch = ch.iloc[:: a.every]
        if cols_wanted:
            ch = ch[cols_wanted]
        if a.max_rows and kept + len(ch) > a.max_rows:
            ch = ch.iloc[: a.max_rows - kept]
        if len(ch):
            ch.to_csv(out, mode="w" if first else "a", header=first, index=False)
            first = False
            kept += len(ch)
        say(f"  scanned {scanned:,} rows, kept {kept:,}")
        if a.max_rows and kept >= a.max_rows:
            say("  reached --max-rows; stopping early")
            break
    if first:
        say("No rows matched - nothing written.")
        return
    guard_size(out, a.allow_large)
    write_provenance(out, source_info(path), {k: v for k, v in vars(a).items() if k not in ("func", "cmd")}, {"rows_scanned": scanned, "rows_kept": kept})
    register_manifest(out.parent, [out], str(path), "table")
    say(f"wrote {out} ({kept:,} rows, {out.stat().st_size / 1e6:.2f} MB)")


# ============================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(prog="datakit", description="Extract only what we need from multi-GB datasets.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="describe a file without loading it")
    p.add_argument("file")
    p.add_argument("--sites")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("auto", help="identify a file and suggest the right command")
    p.add_argument("file")
    p.add_argument("--sites")
    p.set_defaults(func=cmd_auto)

    p = sub.add_parser("dem", help="LOLA or any DEM: horizon masks, terrain/slope grids, small cutouts")
    p.add_argument("--dem", action="append", required=True, help="DEM path; repeat. Horizon bands must be ordered finest/near first; terrain chooses the finest full-footprint DEM per site.")
    p.add_argument("--sites", required=True)
    p.add_argument("--site", action="append", help="only these site ids")
    p.add_argument("--products", default="terrain,horizon", help="comma list of: cutout, terrain, horizon")
    p.add_argument("--radius", type=float, default=2500.0, help="metres: terrain/cutout grid half-width (and default first-band distance)")
    p.add_argument("--footprint-radius", type=float, help="metres: landing-screening radius inside the terrain grid; defaults to --radius")
    p.add_argument("--observer-height", type=float, default=2.0)
    p.add_argument("--azimuth-step", type=float, default=1.0)
    p.add_argument("--max-grid", type=int, default=512)
    p.add_argument("--scale", type=float)
    p.add_argument("--offset", type=float)
    p.add_argument("--count", action="append", default=[], help="companion LOLA count raster; repeat for regional/site products")
    p.add_argument("--elev-uncertainty", action="append", default=[], help="companion elevation-uncertainty raster in metres; repeat")
    p.add_argument("--slope-uncertainty", action="append", default=[], help="companion slope-uncertainty raster in degrees; repeat")
    p.add_argument("--slope-check", action="append", default=[], help="independent slope raster for cross-checking DEM-derived slope; repeat")
    p.add_argument("--slope-check-units", choices=["degrees", "percent"], help="required when --slope-check is supplied")
    p.add_argument("--allow-assumed-scaling", action="store_true",
                   help="allow scale=1/offset=0 when no GDAL/PDS scaling metadata exists; use only after verifying source units")
    p.add_argument("--assume-crs", help="proj4 string, only if the file has no CRS")
    p.add_argument("--out", default="derived")
    p.add_argument("--cutout-dir", default="cutouts", help="cutouts are big: keep them out of git")
    p.add_argument("--allow-large", action="store_true")
    p.set_defaults(func=cmd_dem)

    p = sub.add_parser("spice", help="SPICE kernels -> small ephemeris / site-geometry tables")
    p.add_argument("--kernels", nargs="+", required=True, help="kernel files and/or folders")
    p.add_argument("--start", required=True, help="UTC, e.g. 2026-10-15T00:00:00")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--days", type=float)
    g.add_argument("--end")
    p.add_argument("--step-min", type=float, default=10.0)
    p.add_argument("--bodies", nargs="+", default=["SUN", "EARTH"])
    p.add_argument("--observer", default="MOON")
    p.add_argument("--frame", default="MOON_ME", help="body-fixed frame defined by the loaded FK, e.g. MOON_ME or MOON_ME_DE421")
    p.add_argument("--abcorr", default="LT+S")
    p.add_argument("--sites")
    p.add_argument("--site", action="append")
    p.add_argument("--out", default="derived")
    p.add_argument("--allow-large", action="store_true")
    p.set_defaults(func=cmd_spice)

    p = sub.add_parser("table", help="big CSV/TSV/Parquet: keep only the columns/rows we need")
    p.add_argument("file")
    p.add_argument("--columns")
    p.add_argument("--where", help="pandas query, e.g. \"lat < -80 and quality == 1\"")
    p.add_argument("--time-col")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                   help="e.g. --bbox -89 -88 0 10  (negative latitudes are fine)")
    p.add_argument("--lat-col", default="lat")
    p.add_argument("--lon-col", default="lon")
    p.add_argument("--every", type=int, default=1, help="keep every Nth surviving row")
    p.add_argument("--max-rows", type=int)
    p.add_argument("--chunk", type=int, default=200_000)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-large", action="store_true")
    p.set_defaults(func=cmd_table)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        a.func(a)
    except BrokenPipeError:          # e.g. `datakit ... | head`
        sys.exit(0)


if __name__ == "__main__":
    main()
