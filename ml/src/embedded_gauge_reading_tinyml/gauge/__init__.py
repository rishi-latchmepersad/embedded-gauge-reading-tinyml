"""Gauge calibration and processing public API."""

# Re-export the stable interfaces so imports stay clean.

from .processing import (  # Re-export gauge utilities from processing module.
    GaugeSpec,  # Share the gauge spec dataclass at package level.
    angle_rad_to_fraction,
    fraction_to_angle_rad,
    fraction_to_value,
    load_gauge_specs,  # Share the TOML loader at package level.
    needle_angle_clockwise_rad,  # Share raw angle computation helper.
    needle_fraction,  # Share normalized sweep fraction helper.
    needle_value,  # Share value conversion helper.
    value_to_fraction,
    CALIBRATION_TOML_PATH,
)

__all__ = [  # Define the public symbols for this package.
    "CALIBRATION_TOML_PATH",  # Default TOML path for calibration parameters.
    "GaugeSpec",  # Dataclass describing gauge calibration.
    "angle_rad_to_fraction",
    "fraction_to_angle_rad",
    "fraction_to_value",
    "load_gauge_specs",  # Function to read calibration from TOML.
    "needle_angle_clockwise_rad",  # Raw angle helper.
    "needle_fraction",  # Normalized fraction helper.
    "needle_value",  # Value scaling helper.
    "value_to_fraction",
]
