"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(RuntimeError):
    """Raised when a pipeline configuration is invalid."""


def load_configuration(path: Path) -> dict[str, Any]:
    """Load and perform basic checks on a dataset YAML configuration."""

    path = path.resolve()

    if not path.is_file():
        raise ConfigurationError(
            f"Configuration file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)

    if not isinstance(configuration, dict):
        raise ConfigurationError(
            "Configuration root must be a dictionary."
        )

    required_sections = {
        "schema_version",
        "dataset",
        "cameras",
        "board",
        "frame_selection",
        "detection",
        "calibration",
        "storage",
    }

    missing_sections = required_sections - set(configuration)

    if missing_sections:
        raise ConfigurationError(
            "Configuration is missing sections: "
            f"{sorted(missing_sections)}"
        )

    cameras = configuration["cameras"]

    if not isinstance(cameras, list) or len(cameras) != 2:
        raise ConfigurationError(
            "The current stereo pipeline requires exactly two cameras."
        )

    return configuration
