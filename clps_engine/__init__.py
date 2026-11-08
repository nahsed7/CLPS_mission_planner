"""CLPS New — processable lunar site evaluation engine.

Turn any sites / horizon / geometry / terrain dataset into solar, DTE,
power, and terrain assessments. Not a fixed four-site report.
"""

from .pipeline import MissionPipeline, ScenarioResult
from .sites import Site, load_sites_csv, curated_clps_sites

__all__ = [
    "MissionPipeline",
    "ScenarioResult",
    "Site",
    "load_sites_csv",
    "curated_clps_sites",
]

__version__ = "0.2.0"
