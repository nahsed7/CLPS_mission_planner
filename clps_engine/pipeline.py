"""
MissionPipeline — the processing device.

Modes:
  * live     — Skyfield/DE421 ephemeris
  * offline  — Eshaal Model 1+2 CSVs (no kernels)
  * upload   — user geometry / datakit NPZ
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from . import dte as dte_mod
from . import geometry as geo_mod
from . import offline as offline_mod
from . import power as power_mod
from . import rf as rf_mod
from . import solar as solar_mod
from . import terrain as terrain_mod
from .horizon import HorizonMask, load_horizon_for_site, load_horizon_table, mask_for_site
from .sites import Site
from .windows import summarise_flags


@dataclass
class ScenarioResult:
    site: Site
    timeline: pd.DataFrame
    step_min: float
    has_horizon_mask: bool
    solar: dict
    dte: dict
    power: dict
    rf: dict = field(default_factory=dict)
    terrain: dict = field(default_factory=dict)
    mode: str = "live"
    solar_windows: Optional[pd.DataFrame] = None
    dte_windows: Optional[pd.DataFrame] = None

    def summary_row(self) -> dict:
        row = {
            "site_id": self.site.id,
            "site_name": self.site.name,
            "lat": self.site.lat,
            "lon": self.site.lon,
            "elev_m": self.site.elev_m,
            "mode": self.mode,
            "horizon": "mask" if self.has_horizon_mask else "flat/offline",
            **{f"solar_{k}": v for k, v in self.solar.items() if not isinstance(v, (dict, list))},
            **{f"dte_{k}": v for k, v in self.dte.items() if not isinstance(v, (dict, list))},
            "power_mean_w": self.power.get("mean_power_w"),
            "power_peak_w": self.power.get("peak_power_w"),
            "power_total_energy_kwh": self.power.get("total_energy_kwh"),
            "power_longest_zero_h": self.power.get("longest_zero_power_h"),
            "power_min_battery_soc_pct": self.power.get("min_battery_soc_pct"),
            "power_viable": self.power.get("viable_for_load"),
            "rf_geometry_ok_pct": self.rf.get("geometry_ok_pct"),
            "rf_good_pct": self.rf.get("good_pct"),
            "rf_longest_outage_h": self.rf.get("longest_outage_h"),
            "sun_and_earth_pct": summarise_flags(
                self.timeline, "sunlit", "earth_visible", self.step_min
            )["sun_and_earth_pct"],
        }
        for k, v in self.terrain.items():
            if not isinstance(v, (dict, list)):
                row[f"terrain_{k}"] = v
        return row


class MissionPipeline:
    def __init__(
        self,
        *,
        horizon_dirs: Optional[list] = None,
        terrain_dirs: Optional[list] = None,
        kernel_dir: Optional[Path] = None,
        earth_rule: str = "any_part",
        array: Optional[power_mod.ArrayConfig] = None,
        sun_disc_radius_deg: float = 0.0,
        link: Optional[rf_mod.LinkConfig] = None,
        mode: str = "live",  # live | offline
    ):
        root = Path(__file__).resolve().parent.parent
        self.horizon_dirs = [
            Path(d)
            for d in (
                horizon_dirs
                or [
                    root / "data" / "horizons",
                    root / "data",
                    root.parent / "lunar-mission-browser-main" / "horizon",
                ]
            )
        ]
        self.terrain_dirs = [Path(d) for d in (terrain_dirs or [root / "data" / "terrain"])]
        self.kernel_dir = Path(kernel_dir) if kernel_dir else root / "kernels"
        self.earth_rule = earth_rule
        self.array = array or power_mod.ArrayConfig()
        self.sun_disc_radius_deg = sun_disc_radius_deg
        self.link = link or rf_mod.LinkConfig()
        self.mode = mode
        self._horizon_upload: Optional[pd.DataFrame] = None
        self._geometry_upload = None
        self._geometry_npz: Optional[Path] = None

    def feed_horizons(self, path_or_buf) -> "MissionPipeline":
        self._horizon_upload = load_horizon_table(path_or_buf)
        return self

    def feed_geometry(self, path_or_buf) -> "MissionPipeline":
        self._geometry_upload = path_or_buf
        return self

    def feed_geometry_npz(self, path: Union[str, Path]) -> "MissionPipeline":
        self._geometry_npz = Path(path)
        return self

    def resolve_horizon(self, site: Site) -> Optional[HorizonMask]:
        if self._horizon_upload is not None:
            m = mask_for_site(self._horizon_upload, site.id)
            if m is not None:
                return m
            if "site_id" not in self._horizon_upload.columns:
                return mask_for_site(self._horizon_upload, None)
        return load_horizon_for_site(site.id, self.horizon_dirs)

    def resolve_terrain(self, site: Site) -> dict:
        for d in self.terrain_dirs:
            tr = terrain_mod.load_terrain_product(d, site.id.lower()) or terrain_mod.load_terrain_product(d, site.id)
            if tr is not None:
                return terrain_mod.assess_terrain_grid(tr)
        # Fall back to Model 3 CSV summary
        try:
            m3 = offline_mod.load_model3()
            if site.id in m3.index:
                r = m3.loc[site.id]
                return {
                    "source": "model3_csv",
                    "pct_slope_le_5deg": float(r["pct_slope_le_5deg"]),
                    "pct_slope_le_10deg": float(r["pct_slope_le_10deg"]),
                    "flat_circle_le_5deg_m": float(r["largest_flat_circle_le_5deg_diameter_m"]),
                    "flat_circle_le_10deg_m": float(r["largest_flat_circle_le_10deg_diameter_m"]),
                    "relief_m": float(r["relief_m"]),
                    "mean_slope_deg": float(r["mean_slope_deg"]),
                    "terrain_score": float(r["terrain_score_provisional"]),
                    "terrain_rank": int(r["terrain_rank_provisional"]),
                    "site_elevation_m": float(r["site_elevation_m"]),
                    "coverage_pct": float(r["coverage_pct"]),
                    "source": "model3_csv",
                    "data_quality": str(r.get("data_quality", "legacy_summary; raw grid not packaged")),
                }
                for column in (
                    "site_slope_deg", "roughness_median_m", "roughness_p90_m",
                    "mean_aspect_deg", "count_median",
                    "elevation_uncertainty_m_median", "slope_uncertainty_deg_median",
                    "slope_check_mean_abs_difference_deg",
                ):
                    if column in r.index and pd.notna(r[column]):
                        result[column] = float(r[column])
                if "terrain_source" in r.index:
                    result["dem_file"] = str(r["terrain_source"])
        except FileNotFoundError:
            pass
        return {}

    def _finalize(self, site: Site, geom: pd.DataFrame, mask, mode: str, recompute_flags: bool) -> ScenarioResult:
        step = geo_mod.step_minutes_of(geom) if "time_utc" in geom.columns else 10.0
        if recompute_flags:
            geom = solar_mod.apply_solar_visibility(geom, mask, self.sun_disc_radius_deg)
            geom = dte_mod.apply_dte_visibility(geom, mask, self.earth_rule)
        elif "sunlit" not in geom.columns or "earth_visible" not in geom.columns:
            geom = solar_mod.apply_solar_visibility(geom, mask, self.sun_disc_radius_deg)
            geom = dte_mod.apply_dte_visibility(geom, mask, self.earth_rule)

        geom = power_mod.apply_power(geom, self.array, step)
        geom = rf_mod.apply_rf_quality(geom, self.link)

        return ScenarioResult(
            site=site,
            timeline=geom,
            step_min=step,
            has_horizon_mask=mask is not None,
            solar=solar_mod.solar_report(geom, step),
            dte=dte_mod.dte_report(geom, step),
            power=power_mod.power_report(geom, self.array, step),
            rf=rf_mod.rf_report(geom, step, self.link),
            terrain=self.resolve_terrain(site),
            mode=mode,
            solar_windows=solar_mod.solar_windows(geom, step),
            dte_windows=dte_mod.dte_windows(geom, step),
        )

    def process_site(
        self,
        site: Site,
        *,
        start_iso: Optional[str] = None,
        hours: Optional[float] = None,
        step_min: float = 60.0,
        geometry_csv=None,
        geometry_npz: Optional[Path] = None,
        mode: Optional[str] = None,
    ) -> ScenarioResult:
        mode = mode or self.mode
        mask = self.resolve_horizon(site)

        if mode == "offline":
            geom = offline_mod.merge_offline_geometry(site)
            return self._finalize(site, geom, mask, "offline", recompute_flags=False)

        geom = geo_mod.geometry_from_any(
            site,
            start_iso=start_iso,
            hours=hours,
            step_min=step_min,
            geometry_csv=geometry_csv if geometry_csv is not None else self._geometry_upload,
            geometry_npz=geometry_npz or self._geometry_npz,
            kernel_dir=self.kernel_dir,
        )
        used = "upload" if (geometry_csv or self._geometry_upload or geometry_npz or self._geometry_npz) else "live"
        return self._finalize(site, geom, mask, used, recompute_flags=True)

    def process_many(
        self,
        sites: list[Site],
        *,
        start_iso: Optional[str] = None,
        hours: Optional[float] = None,
        step_min: float = 60.0,
        mode: Optional[str] = None,
    ) -> list[ScenarioResult]:
        return [
            self.process_site(s, start_iso=start_iso, hours=hours, step_min=step_min, mode=mode)
            for s in sites
        ]

    def comparison_table(self, results: list[ScenarioResult]) -> pd.DataFrame:
        return pd.DataFrame([r.summary_row() for r in results])

    def save_result(self, result: ScenarioResult, outdir: Union[str, Path]) -> Path:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        sid = result.site.id
        result.timeline.to_csv(outdir / f"{sid}_timeline.csv", index=False)
        if result.solar_windows is not None:
            result.solar_windows.to_csv(outdir / f"{sid}_solar_windows.csv", index=False)
        if result.dte_windows is not None:
            result.dte_windows.to_csv(outdir / f"{sid}_dte_windows.csv", index=False)
        meta = {
            "site": result.site.to_dict(),
            "mode": result.mode,
            "step_min": result.step_min,
            "has_horizon_mask": result.has_horizon_mask,
            "solar": result.solar,
            "dte": result.dte,
            "power": {k: v for k, v in result.power.items() if k != "array"},
            "array": result.power.get("array"),
            "rf": {k: v for k, v in result.rf.items() if k != "config"},
            "terrain": result.terrain,
        }
        (outdir / f"{sid}_summary.json").write_text(json.dumps(meta, indent=2, default=str))
        return outdir

    def save_comparison(self, results: list[ScenarioResult], outdir: Union[str, Path]) -> Path:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        table = self.comparison_table(results)
        path = outdir / "site_comparison.csv"
        table.to_csv(path, index=False)
        for r in results:
            self.save_result(r, outdir)
        return path
