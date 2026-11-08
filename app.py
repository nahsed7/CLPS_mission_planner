"""
CLPS New — complete Lunar South Pole Mission Planner
NASA Space Apps Challenge 2026
"""
from __future__ import annotations

import base64
import json
import sys
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from clps_engine.dte import EARTH_RULES
from clps_engine.horizon import horizon_at
from clps_engine.offline import offline_team_scorecard
from clps_engine.pipeline import MissionPipeline
from clps_engine.power import ArrayConfig, compare_array_configs
from clps_engine.sites import curated_clps_sites, custom_site, load_sites_csv, merge_sites
from clps_engine.terrain import load_terrain_product
from clps_engine.windows import time_remaining_hours

st.set_page_config(page_title="CLPS New · Lunar Mission Planner", page_icon="🌑", layout="wide")

DATA = ROOT / "data"
OUT = ROOT / "outputs"
STATIC = ROOT / "static"

MISSION_WINDOW_NOTE = (
    "**Canonical analysis window (Models 1–2):** 2026-10-15 → 2027-01-12 UTC "
    "(**89.06 days ≈ 3 lunar synodic months**, 10-minute steps, 12 824 samples/site). "
    "One lunar day-night cycle is ~29.5 days (~708 h); the default live window can be 1–3 cycles."
)


def _rule_label(key: str) -> str:
    return {
        "any_part": "Any part of Earth's disc (NASA / Model 2)",
        "centre": "Centre of Earth above horizon",
        "whole_disc": "Whole Earth disc above horizon",
    }.get(key, key)


def hillshade(h, mpp, az_deg=315.0, alt_deg=35.0):
    gy, gx = np.gradient(h.astype("float64"), mpp)
    n = np.dstack([-gx, gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    light = np.array([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])
    return np.clip(n @ light, 0, 1)


def render_3d(df, site_name, mask, has_mask, step_min, tr=None, height=640):
    path = STATIC / "viewer3d.html"
    if not path.exists():
        st.warning("viewer3d.html missing in static/")
        return
    k = max(1, int(np.ceil(len(df) / 1500)))
    d = df.iloc[::k]
    r = lambda a, n=3: [round(float(x), n) for x in a]
    # Map column names
    sun_az = d["sun_az"] if "sun_az" in d else d.get("Sun_Az", pd.Series([0] * len(d)))
    sun_alt = d["sun_alt"] if "sun_alt" in d else d.get("Sun_Alt", pd.Series([0] * len(d)))
    earth_az = d["earth_az"] if "earth_az" in d else d.get("Earth_Az", pd.Series([0] * len(d)))
    earth_alt = d["earth_alt"] if "earth_alt" in d else d.get("Earth_Alt", pd.Series([0] * len(d)))
    payload = {
        "site": site_name,
        "stepMin": step_min * k,
        "times": [f"{x:%Y-%m-%d %H:%M} UTC" for x in pd.to_datetime(d["time_utc"])],
        "sunAz": r(sun_az),
        "sunAlt": r(sun_alt),
        "earthAz": r(earth_az),
        "earthAlt": r(earth_alt),
        "sunlit": [int(x) for x in d["sunlit"]],
        "earthVis": [int(x) for x in d["earth_visible"]],
        "horizon": r(horizon_at(mask, np.arange(360, dtype=float)), 2) if has_mask else None,
        "horizonSource": "mask" if has_mask else None,
        "terrain": None,
    }
    if tr is not None:
        hgt = tr["height"]
        valid = np.isfinite(hgt) & tr.get("valid_mask", np.isfinite(hgt))
        scale = float(max(np.abs(hgt[valid]).max(), 1.0) / 32000.0) if valid.any() else 1.0
        ints = np.full(hgt.shape, -32768, dtype="<i2")
        ints[valid] = np.clip(np.round(hgt[valid] / scale), -32767, 32767).astype("<i2")
        meta = tr["meta"]
        lon = np.radians(meta["lon"])
        north = meta.get("north_xy", [float(np.sin(lon)), float(np.cos(lon))])
        east = meta.get("east_xy", [float(np.cos(lon)), float(-np.sin(lon))])
        payload["terrain"] = {
            "rows": int(hgt.shape[0]),
            "cols": int(hgt.shape[1]),
            "mpp": float(meta["metres_per_pixel"]),
            "sitePixel": meta.get("site_pixel", [hgt.shape[1] / 2, hgt.shape[0] / 2]),
            "demFile": meta.get("dem_file", "source DEM"),
            "sourceMpp": meta.get("source_metres_per_pixel", meta.get("metres_per_pixel")),
            "scale": scale,
            "north": north,
            "east": east,
            "b64": base64.b64encode(ints.tobytes()).decode("ascii"),
            "validB64": base64.b64encode(valid.astype("uint8").tobytes()).decode("ascii"),
        }
    html = path.read_text(encoding="utf-8").replace("__DATA__", json.dumps(payload))
    components.html(html, height=height, scrolling=False)


# --------------------------------------------------------------------------- UI
st.title("CLPS New — Lunar South Pole Mission Planner")
st.caption("Rapid site–date evaluation for solar power and direct-to-Earth windows · NASA Space Apps 2026")
st.info(MISSION_WINDOW_NOTE)

st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Computation mode",
    ["offline", "live"],
    format_func=lambda m: {
        "offline": "Offline — use team Model 1+2 CSVs (no kernels, ~89 days)",
        "live": "Live — Skyfield/DE421 (downloads kernels once)",
    }[m],
    index=0,
)

st.sidebar.header("Sites")
use_curated = st.sidebar.checkbox("Curated CLPS / south-pole sites", True)
sites_file = st.sidebar.file_uploader("Upload sites CSV", type=["csv", "tsv"])
custom_on = st.sidebar.checkbox("Custom coordinates")
if custom_on:
    clat = st.sidebar.number_input("Lat (°)", -89.5, min_value=-90.0, max_value=90.0, format="%.6f")
    clon = st.sidebar.number_input("Lon (°E)", 0.0, min_value=-180.0, max_value=180.0, format="%.6f")
    celev = st.sidebar.number_input("Elev (m)", 0.0, format="%.1f")
    cname = st.sidebar.text_input("Name", "Custom site")

st.sidebar.header("Optional data feeds")
horizon_file = st.sidebar.file_uploader("Horizon CSV", type=["csv"])
geometry_file = st.sidebar.file_uploader("Geometry CSV (live/upload)", type=["csv"])

if mode == "live":
    st.sidebar.header("Live time window")
    start_date = st.sidebar.date_input("Start (UTC)", date(2026, 10, 15), min_value=date(1990, 1, 1), max_value=date(2049, 12, 31))
    start_hour = st.sidebar.slider("Start hour", 0, 23, 0)
    duration_h = st.sidebar.slider("Duration (h)", 24, 2160, 708, step=24, help="708 h ≈ 1 lunar day; 2136 h ≈ 3 lunar days (team window)")
    step_min = st.sidebar.select_slider("Step (min)", [10, 15, 30, 60], 60)
else:
    start_date, start_hour, duration_h, step_min = date(2026, 10, 15), 0, 2136, 10
    st.sidebar.caption("Offline mode uses the full 89.06-day team timelines (10 min steps).")

st.sidebar.header("Earth & power")
earth_rule = st.sidebar.selectbox("Earth visibility", list(EARTH_RULES), format_func=_rule_label, index=0)
nameplate = st.sidebar.number_input("Array nameplate (W)", 50.0, 5000.0, 500.0, 50.0)
tracking = st.sidebar.selectbox("Tracking", ["dual_axis", "vertical_axis", "fixed"])
tilt = st.sidebar.slider("Tilt (°)", 0.0, 90.0, 0.0)
load_w = st.sidebar.number_input("Load (W)", 1.0, 2000.0, 50.0, 5.0)
battery_wh = st.sidebar.number_input("Battery (Wh)", 100.0, 50000.0, 3000.0, 100.0)

registry = {}
if use_curated:
    registry = merge_sites([s for s in curated_clps_sites() if s.id in ("LM1", "Site06", "Site20", "Site23") or mode == "live"])
if sites_file is not None:
    registry = merge_sites(registry.values(), load_sites_csv(sites_file))
if custom_on:
    registry["CUSTOM"] = custom_site(clat, clon, celev, cname)

if not registry:
    st.warning("Add at least one site.")
    st.stop()

# Prefer the four mission sites first
default_ids = [i for i in ("LM1", "Site06", "Site20", "Site23") if i in registry] or list(registry)[:2]
chosen_ids = st.multiselect(
    "Compare sites",
    list(registry.keys()),
    default=default_ids,
    format_func=lambda i: f"{registry[i].name} ({i})",
)
if not chosen_ids:
    st.stop()

array = ArrayConfig(
    nameplate_w=float(nameplate),
    tracking=tracking,
    tilt_deg=float(tilt),
    night_draw_w=float(load_w),
    battery_wh=float(battery_wh),
)
pipe = MissionPipeline(
    horizon_dirs=[DATA / "horizons", DATA, ROOT.parent / "lunar-mission-browser-main" / "horizon"],
    terrain_dirs=[DATA / "terrain"],
    kernel_dir=ROOT / "kernels",
    earth_rule=earth_rule,
    array=array,
    mode=mode,
)
if horizon_file is not None:
    pipe.feed_horizons(horizon_file)
if geometry_file is not None:
    pipe.feed_geometry(geometry_file)
    mode = "live"

start_iso = datetime.combine(start_date, time(int(start_hour))).isoformat()

col_run, col_score = st.columns([1, 1])
run = col_run.button("▶ Run mission evaluation", type="primary")
show_team = col_score.checkbox("Show team Model 1–3 scorecard", value=True)

if show_team:
    with st.expander("Team scorecard (Eshaal Models 1–3, packaged CSVs)", expanded=not run and "last_results" not in st.session_state):
        try:
            sc = offline_team_scorecard()
            st.dataframe(sc.set_index("site_id"), use_container_width=True)
            fig = go.Figure()
            for col, name in [("sunlit_pct", "Sunlit %"), ("earth_visible_pct", "Earth %")]:
                if col in sc.columns:
                    fig.add_bar(x=sc["site_name"], y=sc[col], name=name)
            fig.update_layout(template="plotly_dark", barmode="group", height=320, yaxis_title="%", margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(MISSION_WINDOW_NOTE)
        except Exception as e:
            st.warning(f"Team scorecard unavailable: {e}")

if not run and "last_results" not in st.session_state:
    st.markdown(
        """
### Product status
| Capability | Status |
|---|---|
| Model 1 solar + Model 2 DTE (team CSVs) | ✅ Offline mode |
| Live Skyfield geometry | ✅ Live mode |
| Power generation + battery SoC | ✅ |
| Geometric RF quality | ✅ |
| Terrain (Model 3 DEM products) | ✅ |
| Custom sites / uploads | ✅ |
| 3D viewer | ✅ (uses each site's processed DEM when available) |
| Validation suite | ✅ `tests/validate_project.py` |
"""
    )
    st.stop()

if run:
    with st.spinner("Processing sites…"):
        try:
            sites = [registry[i] for i in chosen_ids]
            if mode == "offline":
                # Custom sites without offline CSVs fall back to live if possible
                results = []
                for s in sites:
                    try:
                        results.append(pipe.process_site(s, mode="offline"))
                    except FileNotFoundError:
                        st.warning(f"{s.id}: no offline CSV — skipping (use live mode or upload geometry).")
            else:
                results = pipe.process_many(
                    sites, start_iso=start_iso, hours=float(duration_h), step_min=float(step_min), mode="live"
                )
            if not results:
                st.error("No sites processed.")
                st.stop()
            st.session_state["last_results"] = results
            stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            st.session_state["last_out"] = str(pipe.save_comparison(results, OUT / stamp))
        except Exception as e:
            st.error(f"Failed: {e}")
            st.exception(e)
            st.stop()

results = st.session_state["last_results"]
pipe_view = pipe
table = pipe_view.comparison_table(results)

st.subheader("Comparison")
prefer = [
    "site_id", "site_name", "mode", "horizon",
    "solar_sunlit_pct", "dte_earth_visible_pct", "sun_and_earth_pct",
    "power_mean_w", "power_total_energy_kwh", "power_longest_zero_h", "power_min_battery_soc_pct", "power_viable",
    "rf_geometry_ok_pct", "rf_good_pct", "rf_longest_outage_h",
    "terrain_terrain_score", "terrain_pct_slope_le_10deg", "terrain_site_slope_deg",
]
cols = [c for c in prefer if c in table.columns]
st.dataframe(table[cols].set_index("site_id"), use_container_width=True)
st.download_button("Download comparison CSV", table.to_csv(index=False).encode(), "clpsnew_comparison.csv", "text/csv")
if "last_out" in st.session_state:
    st.caption(f"Saved → `{st.session_state['last_out']}`")

focus = st.selectbox("Inspect", [r.site.id for r in results], format_func=lambda i: next(r.site.name for r in results if r.site.id == i))
result = next(r for r in results if r.site.id == focus)
df = result.timeline
step = result.step_min
mask = pipe.resolve_horizon(result.site)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sunlit", f"{result.solar['sunlit_pct']:.1f}%")
m2.metric("Earth / DTE", f"{result.dte['earth_visible_pct']:.1f}%")
m3.metric("Mean power", f"{result.power['mean_power_w']:.0f} W")
m4.metric("Battery floor", f"{result.power['min_battery_soc_pct']:.0f}%")
m5.metric("RF good", f"{result.rf.get('good_pct', 0):.1f}%")
m6.metric("Sun+Earth", f"{result.summary_row()['sun_and_earth_pct']:.1f}%")

tabs = st.tabs(
    ["⚡ Power", "📡 DTE / RF", "☀️ Solar windows", "⏱ Operations clock", "📈 Elevations", "🌑 3D", "⛰ Terrain", "ℹ️ Assumptions"]
)

with tabs[0]:
    st.markdown(
        f"Array **{nameplate:.0f} W** · `{tracking}` · load **{load_w:.0f} W** · battery **{battery_wh:.0f} Wh** · "
        f"{'✅ survives continuous load' if result.power['viable_for_load'] else '❌ battery depleted in window'}"
    )
    if result.mode == "offline":
        st.caption("Offline power uses sunlit flags from Model 1 with a 3° elevation proxy when lit (dual-axis recommended).")
    figp = go.Figure()
    figp.add_trace(go.Scatter(x=df["time_utc"], y=df["power_w"], name="Array W", line=dict(color="#F0C040")))
    figp.add_trace(go.Scatter(x=df["time_utc"], y=df["load_w"], name="Load W", line=dict(color="#E74C3C", dash="dash")))
    figp.add_trace(go.Scatter(x=df["time_utc"], y=df["battery_soc_pct"], name="SoC %", yaxis="y2", line=dict(color="#2ECC71")))
    figp.update_layout(
        template="plotly_dark", height=400,
        yaxis=dict(title="W"), yaxis2=dict(title="SoC %", overlaying="y", side="right", range=[0, 105]),
        legend=dict(orientation="h", y=-0.2), margin=dict(t=20),
    )
    st.plotly_chart(figp, use_container_width=True)
    base = df[["time_utc", "sun_alt", "sunlit"]].copy()
    st.dataframe(
        compare_array_configs(
            base,
            [
                ArrayConfig(nameplate_w=nameplate, tracking="fixed", tilt_deg=0, night_draw_w=load_w, battery_wh=battery_wh),
                ArrayConfig(nameplate_w=nameplate, tracking="fixed", tilt_deg=30, night_draw_w=load_w, battery_wh=battery_wh),
                ArrayConfig(nameplate_w=nameplate, tracking="vertical_axis", tilt_deg=tilt, night_draw_w=load_w, battery_wh=battery_wh),
                ArrayConfig(nameplate_w=nameplate, tracking="dual_axis", night_draw_w=load_w, battery_wh=battery_wh),
            ],
            step,
        ),
        use_container_width=True,
    )

with tabs[1]:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Visible", f"{result.dte['earth_visible_pct']:.1f}%")
    d2.metric("Longest blackout", f"{result.dte['longest_dte_blackout_h']:.0f} h")
    d3.metric("RF geometry OK", f"{result.rf.get('geometry_ok_pct', 0):.1f}%")
    d4.metric("Longest RF outage", f"{result.rf.get('longest_outage_h', 0):.0f} h")
    if "visibility_margin_deg" in df.columns:
        figd = go.Figure()
        figd.add_hline(y=0, line_dash="dash", line_color="red")
        figd.add_trace(go.Scatter(x=df["time_utc"], y=df["visibility_margin_deg"], fill="tozeroy", name="Margin", line=dict(color="#1E90FF")))
        if "rf_quality" in df.columns:
            figd.add_trace(go.Scatter(x=df["time_utc"], y=df["rf_quality"], name="RF quality 0–1", yaxis="y2", line=dict(color="#9B59B6")))
            figd.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1.05], title="Quality"))
        figd.update_layout(template="plotly_dark", height=360, yaxis_title="Margin (°)", margin=dict(t=20))
        st.plotly_chart(figd, use_container_width=True)
    st.dataframe(result.dte_windows, use_container_width=True, height=260)
    st.caption(result.rf.get("note", ""))

with tabs[2]:
    s1, s2, s3 = st.columns(3)
    s1.metric("Longest sunlight", f"{result.solar['longest_sunlight_h']:.0f} h")
    s2.metric("Longest darkness", f"{result.solar['longest_darkness_h']:.0f} h")
    s3.metric("Max Sun alt", f"{result.solar.get('max_sun_alt_deg', float('nan')):.2f}°")
    st.dataframe(result.solar_windows, use_container_width=True, height=280)

with tabs[3]:
    idx = st.slider("Time index", 0, max(0, len(df) - 1), 0, key="ops_idx")
    row = df.iloc[idx]
    st.markdown(f"**{row['time_utc']} UTC**")
    sun = df["sunlit"].to_numpy(bool)
    earth = df["earth_visible"].to_numpy(bool)
    c1, c2, c3, c4 = st.columns(4)
    if sun[idx]:
        c1.success(f"☀️ Sun up · {time_remaining_hours(sun, idx, step):.1f} h to sunset")
    else:
        c1.error(f"🌑 Dark · {time_remaining_hours(~sun, idx, step):.1f} h to sunrise")
    if earth[idx]:
        c2.success(f"🌍 Earth up · {time_remaining_hours(earth, idx, step):.1f} h to Earthset")
    else:
        c2.error(f"📡 No DTE · {time_remaining_hours(~earth, idx, step):.1f} h to Earthrise")
    c3.metric("Power now", f"{row['power_w']:.0f} W")
    c4.metric("Battery", f"{row['battery_soc_pct']:.0f}%")

with tabs[4]:
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.add_trace(go.Scatter(x=df["time_utc"], y=df["sun_alt"], name="Sun elev", line=dict(color="#FFD700")))
    fig.add_trace(go.Scatter(x=df["time_utc"], y=df["earth_alt"], name="Earth elev", line=dict(color="#1E90FF")))
    if "sun_horizon_deg" in df.columns:
        fig.add_trace(go.Scatter(x=df["time_utc"], y=df["sun_horizon_deg"], name="Horizon→Sun", line=dict(color="#FFD700", dash="dot")))
    if "earth_horizon_deg" in df.columns:
        fig.add_trace(go.Scatter(x=df["time_utc"], y=df["earth_horizon_deg"], name="Horizon→Earth", line=dict(color="#1E90FF", dash="dot")))
    fig.update_layout(template="plotly_dark", height=400, hovermode="x unified", yaxis_title="Elevation (°)")
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    tr = load_terrain_product(DATA / "terrain", result.site.id.lower())
    if tr is None:
        st.caption("No processed DEM grid is installed for this site yet. The 3D terrain view requires a validated Model 3 terrain product.")
    render_3d(df, result.site.name, mask, mask is not None, step, tr)

with tabs[6]:
    if result.terrain:
        keys = [
            ("terrain_score", "Score"),
            ("pct_slope_le_10deg", "Area ≤10° %"),
            ("flat_circle_le_10deg_m", "Flat pad (m)"),
            ("relief_m", "Relief (m)"),
            ("site_slope_deg", "Site slope"),
        ]
        cols = st.columns(len(keys))
        for col, (k, label) in zip(cols, keys):
            if k in result.terrain:
                v = result.terrain[k]
                col.metric(label, f"{v:.1f}" if isinstance(v, float) else v)
        if result.terrain.get("source") == "model3_csv":
            st.caption("Stats from the Model 3 comparison CSV. Check each row's DEM source, cell size and valid coverage; this summary is not a substitute for a loaded per-pixel grid.")
        tr = load_terrain_product(DATA / "terrain", result.site.id.lower())
        if tr is not None:
            meta, hgt, slp = tr["meta"], tr["height"], tr["slope"]
            mpp = meta["metres_per_pixel"]
            shade = np.ma.masked_invalid(hillshade(hgt, mpp))
            rows, cols_n = hgt.shape
            site_px = meta.get("site_pixel", [cols_n / 2, rows / 2])
            xs = (np.arange(cols_n) - site_px[0]) * mpp
            ys = (site_px[1] - np.arange(rows)) * mpp
            fig = go.Figure(go.Heatmap(x=xs, y=ys[::-1], z=shade[::-1], colorscale="gray", showscale=False))
            fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="x", size=12, color="cyan"), name="Site"))
            fig.update_layout(template="plotly_dark", height=480, title=f"{result.site.name} · DEM hillshade", margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        img = STATIC / "slope_maps_all_sites.png"
        if img.exists():
            st.image(str(img), caption="Team Model 3 slope maps (all sites)")
    else:
        st.info("No terrain data for this site.")

with tabs[7]:
    st.markdown(
        f"""
- **Mode:** `{result.mode}` · **Earth rule:** {_rule_label(earth_rule)}
- **Horizon:** {"terrain mask loaded" if result.has_horizon_mask else "flat / offline flags (no 360° mask for this site in package — only LM1 ships `lm1_horizon.csv`)"}
- **Ephemeris (live):** Skyfield + JPL DE421 + lunar PA frame (1900–2050)
- **Offline:** Model 1 sunlit + Model 2 Earth geometry from team SPICE runs
- **Power:** nameplate × incidence cosine × derates; battery = coulomb counter (not thermal EPS)
- **RF:** geometric margin/quality only — **not** a link budget
- **Not modelled:** atmosphere (N/A), boulders, relays, ground-station scheduling, thermal
- {MISSION_WINDOW_NOTE}
"""
    )
