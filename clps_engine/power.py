"""
Power generation module — missing from the original ~60% solution.

Converts solar elevation (and optional array tilt/tracking) into estimated
electrical power for a lander/rover solar array. Works on ANY geometry
timeline that already has sun_alt / sunlit columns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .windows import runs_table

# Lunar solar constant at 1 AU (W/m^2). Same order as Earth's top-of-atmosphere.
LUNAR_SOLAR_CONSTANT_W_M2 = 1361.0


@dataclass
class ArrayConfig:
    """Physical / electrical description of the solar array."""

    nameplate_w: float = 500.0          # peak DC at AM0 / normal incidence
    area_m2: float = 2.0                # used if nameplate not preferred
    efficiency: float = 0.30            # cell+harness efficiency at AM0
    use_nameplate: bool = True          # if True, scale nameplate by cos(incidence)
    tilt_deg: float = 0.0               # fixed tilt from local horizontal toward Sun azimuth plane
    tracking: str = "fixed"             # fixed | vertical_axis | dual_axis
    dust_factor: float = 1.0             # 0..1 optical degradation
    temp_derate: float = 0.95           # simple thermal derate at operating temp
    min_elevation_deg: float = 0.0      # array produces 0 below this solar elevation
    night_draw_w: float = 50.0          # continuous bus load estimate (for energy balance)
    battery_wh: float = 2000.0          # usable battery energy for eclipse sizing

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_ARRAY = ArrayConfig()


def _incidence_cos(sun_alt_deg: np.ndarray, cfg: ArrayConfig) -> np.ndarray:
    """
    Cosine of incidence angle between Sun vector and array normal.

    - dual_axis: always face the Sun when above horizon → cos = 1 when lit
    - vertical_axis: panel normal in local vertical plane toward Sun → cos ≈ cos(zenith - tilt)
    - fixed: flat or fixed-tilt facing local zenith (tilt_deg from horizontal)
    """
    alt = np.radians(np.asarray(sun_alt_deg, dtype=float))
    zenith = np.pi / 2 - alt
    tilt = np.radians(cfg.tilt_deg)

    mode = (cfg.tracking or "fixed").lower()
    if mode == "dual_axis":
        mu = np.ones_like(alt)
    elif mode == "vertical_axis":
        # Track in azimuth; incidence ≈ |zenith - tilt|
        mu = np.cos(zenith - tilt)
    else:
        # Fixed panel: normal tilted from zenith by (90 - tilt) toward a fixed az — we
        # approximate with tilt from horizontal, Sun in the tilt plane (optimistic).
        mu = np.sin(alt + tilt)  # flat tilt=0 → sin(alt) = cos(zenith)
    return np.clip(mu, 0.0, 1.0)


def instantaneous_power_w(
    sun_alt_deg: np.ndarray,
    sunlit: np.ndarray,
    cfg: ArrayConfig = DEFAULT_ARRAY,
) -> np.ndarray:
    """Electrical power (W) at each sample."""
    sun_alt = np.asarray(sun_alt_deg, dtype=float)
    lit = np.asarray(sunlit, dtype=bool)
    mu = _incidence_cos(sun_alt, cfg)
    mu = np.where(sun_alt >= cfg.min_elevation_deg, mu, 0.0)
    mu = np.where(lit, mu, 0.0)

    if cfg.use_nameplate:
        p_peak = cfg.nameplate_w
    else:
        p_peak = LUNAR_SOLAR_CONSTANT_W_M2 * cfg.area_m2 * cfg.efficiency

    scale = cfg.dust_factor * cfg.temp_derate
    return p_peak * mu * scale


def apply_power(
    df: pd.DataFrame,
    cfg: ArrayConfig = DEFAULT_ARRAY,
    step_min: Optional[float] = None,
) -> pd.DataFrame:
    """
    Attach power columns to a solar/DTE timeline.

    Requires: sun_alt, sunlit. Adds power_w, energy_wh_step, cumulative_wh,
    load_w, net_w, battery_soc_wh (simple coulomb-count against night_draw).
    """
    out = df.copy()
    if "sunlit" not in out.columns:
        raise ValueError("apply_power needs a 'sunlit' column — run solar.apply_solar_visibility first")
    if step_min is None:
        from .adapters import infer_step_minutes
        step_min = infer_step_minutes(out["time_utc"])

    p = instantaneous_power_w(out["sun_alt"].to_numpy(), out["sunlit"].to_numpy(), cfg)
    out["power_w"] = p
    out["load_w"] = cfg.night_draw_w
    out["net_w"] = p - cfg.night_draw_w
    hours = step_min / 60.0
    out["energy_wh_step"] = p * hours
    out["cumulative_wh"] = out["energy_wh_step"].cumsum()

    # Simple battery SoC walk (clipped)
    soc = np.zeros(len(out), dtype=float)
    soc0 = cfg.battery_wh  # start full
    level = soc0
    for i, net in enumerate(out["net_w"].to_numpy()):
        level = float(np.clip(level + net * hours, 0.0, cfg.battery_wh))
        soc[i] = level
    out["battery_soc_wh"] = soc
    out["battery_soc_pct"] = 100.0 * soc / max(cfg.battery_wh, 1e-9)
    return out


def power_report(df: pd.DataFrame, cfg: ArrayConfig, step_min: float) -> dict:
    """Mission-planner metrics for the power subsystem."""
    p = df["power_w"].to_numpy(float)
    lit = df["sunlit"].to_numpy(bool) if "sunlit" in df.columns else p > 0
    hours = step_min / 60.0
    energy_total = float(p.sum() * hours)
    duration_h = len(df) * hours
    producing = p > 0.5  # ignore numerical dust

    out = {
        "array": cfg.to_dict(),
        "mean_power_w": float(p.mean()),
        "mean_power_when_lit_w": float(p[lit].mean()) if lit.any() else 0.0,
        "peak_power_w": float(p.max()),
        "total_energy_kwh": energy_total / 1000.0,
        "mean_energy_per_earth_day_kwh": (energy_total / max(duration_h, 1e-9)) * 24.0 / 1000.0,
        "duty_cycle_producing_pct": float(100.0 * producing.mean()),
        "longest_zero_power_h": float(
            runs_table(df["time_utc"], producing, step_min, ("on", "off")).query("state == 'off'")["hours"].max() or 0.0
        )
        if (~producing).any()
        else 0.0,
        "min_battery_soc_pct": float(df["battery_soc_pct"].min()) if "battery_soc_pct" in df.columns else float("nan"),
        "ends_battery_soc_pct": float(df["battery_soc_pct"].iloc[-1]) if "battery_soc_pct" in df.columns else float("nan"),
        "energy_margin_vs_load_kwh": float(((p - cfg.night_draw_w).sum() * hours) / 1000.0),
        "viable_for_load": bool(df["battery_soc_wh"].min() > 0.01 * cfg.battery_wh)
        if "battery_soc_wh" in df.columns
        else bool(p.mean() >= cfg.night_draw_w),
    }
    return out


def compare_array_configs(
    df: pd.DataFrame,
    configs: list[ArrayConfig],
    step_min: float,
) -> pd.DataFrame:
    """Side-by-side power reports for several array designs on the same timeline."""
    rows = []
    for cfg in configs:
        powered = apply_power(df, cfg, step_min)
        rep = power_report(powered, cfg, step_min)
        rows.append(
            {
                "config": cfg.tracking if cfg.nameplate_w else cfg.tracking,
                "nameplate_w": cfg.nameplate_w,
                "tracking": cfg.tracking,
                "tilt_deg": cfg.tilt_deg,
                "mean_power_w": round(rep["mean_power_w"], 1),
                "peak_power_w": round(rep["peak_power_w"], 1),
                "total_energy_kwh": round(rep["total_energy_kwh"], 2),
                "longest_zero_power_h": round(rep["longest_zero_power_h"], 1),
                "min_battery_soc_pct": round(rep["min_battery_soc_pct"], 1),
                "viable_for_load": rep["viable_for_load"],
            }
        )
    return pd.DataFrame(rows)
