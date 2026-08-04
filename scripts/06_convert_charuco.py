#!/usr/bin/env python3
"""Convert raw ArUco Nano detections to CalibCam ChArUco YAML."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SOURCE_ROOT))

from aruconano_calibcam.stages.convert_charuco import (  # noqa: E402
    convert_to_charuco,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one camera's raw ArUco Nano "
            "markers to CalibCam ChArUco detections."
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
        "--marker-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--camera",
        required=True,
        choices=("left", "right"),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--regression-tolerance-px",
        type=float,
        default=0.1,
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    manifest_path = arguments.manifest.resolve()

    if manifest_path.exists():
        raise FileExistsError(
            f"Manifest already exists: "
            f"{manifest_path}"
        )

    report = convert_to_charuco(
        project_root=PROJECT_ROOT,
        configuration_path=arguments.config,
        extraction_manifest_path=(
            arguments.extraction_manifest
        ),
        marker_manifest_path=(
            arguments.marker_manifest
        ),
        camera_name=arguments.camera,
        output_path=arguments.output,
        reference_path=arguments.reference,
        regression_tolerance_px=(
            arguments.regression_tolerance_px
        ),
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )
        handle.write("\n")

    conversion = report["conversion"]

    print()
    print("ChArUco conversion")
    print("==================")
    print(f"Dataset: {report['dataset_id']}")
    print(
        f"Camera:  "
        f"{report['camera']['index']} "
        f"({report['camera']['name']})"
    )
    print(
        "Frames with Nano markers:",
        conversion["frames_with_nano_markers"],
    )
    print(
        "Frames with ChArUco:",
        conversion["frames_with_charuco"],
    )
    print(
        "Valid CalibCam frames:",
        conversion["valid_calibcam_frames"],
    )
    print(
        "Active ChArUco IDs:",
        conversion["active_charuco_ids"],
    )
    print(
        "Mean corners/frame:",
        f"{conversion['mean_corners_per_frame']:.2f}",
    )
    print(
        "Median corners/frame:",
        f"{conversion['median_corners_per_frame']:.2f}",
    )
    print(
        "First/last frame:",
        conversion["first_frame_index"],
        "/",
        conversion["last_frame_index"],
    )
    print(
        "YAML anchors:",
        report["output"][
            "contains_yaml_anchor"
        ],
    )

    regression = report["regression"]

    if regression is not None:
        print(
            "Regression passed:",
            regression["passed"],
        )
        print(
            "Maximum coordinate difference:",
            regression[
                "maximum_coordinate_difference_px"
            ],
            "px",
        )

    print(f"Valid: {report['valid']}")
    print()
    print("Detection YAML:", report["output"]["yaml"])
    print("Manifest:", manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
