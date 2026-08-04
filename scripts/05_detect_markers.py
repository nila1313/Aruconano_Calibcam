#!/usr/bin/env python3
"""Run raw ArUco Nano marker detection for one camera."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SOURCE_ROOT))

from aruconano_calibcam.stages.detect_markers import (  # noqa: E402
    run_marker_detection,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and validate native ArUco Nano "
            "marker detection."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--extraction-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--camera",
        required=True,
        choices=("left", "right"),
    )

    parser.add_argument(
        "--detector",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    manifest_path = arguments.manifest.resolve()

    if manifest_path.exists():
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}"
        )

    report = run_marker_detection(
        project_root=PROJECT_ROOT,
        configuration_path=arguments.config,
        extraction_manifest_path=(
            arguments.extraction_manifest
        ),
        camera_name=arguments.camera,
        detector_binary=arguments.detector,
        output_directory=arguments.output_dir,
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    statistics = report["statistics"]

    print()
    print("Marker-detection validation")
    print("===========================")
    print(f"Dataset: {report['dataset_id']}")
    print(
        f"Camera:  "
        f"{report['camera']['index']} "
        f"({report['camera']['name']})"
    )
    print(
        "Frames processed:",
        statistics["frames_processed"],
    )
    print(
        "Frames with markers:",
        statistics["frames_with_markers"],
    )
    print(
        "Frames without markers:",
        statistics["frames_without_markers"],
    )
    print(
        "Total marker detections:",
        statistics["total_marker_detections"],
    )
    print(
        "Marker IDs:",
        statistics["observed_marker_ids"],
    )
    print(
        "Unexpected marker IDs:",
        statistics["unexpected_marker_ids"],
    )
    print(
        "Duration:",
        report["detector"]["duration_seconds"],
        "seconds",
    )
    print(f"Valid: {report['valid']}")

    for warning in report["warnings"]:
        print(f"[WARNING] {warning}")

    print()
    print(
        "Detections:",
        report["outputs"]["detections_csv"],
    )
    print(
        "Summary:",
        report["outputs"]["summary_csv"],
    )
    print("Manifest:", manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
