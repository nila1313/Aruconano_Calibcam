#!/usr/bin/env python3
"""Generate a selected-frame plan for one dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SOURCE_ROOT))

from aruconano_calibcam.stages.plan_frames import (  # noqa: E402
    build_frame_plan,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan synchronized calibration frames without "
            "extracting images."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Dataset configuration YAML file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON frame plan.",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    plan = build_frame_plan(
        project_root=PROJECT_ROOT,
        configuration_path=arguments.config,
    )

    output_path = arguments.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")

    selection = plan["selection"]

    print("Frame planning")
    print("==============")
    print(f"Dataset:        {plan['dataset_id']}")
    print(f"Start:          {selection['start']}")
    print(
        "Effective end: ",
        selection["effective_end_exclusive"],
        "(exclusive)",
    )
    print(f"Step:           {selection['step']}")
    print(
        f"Selected frames: "
        f"{selection['base_selected_count']}"
    )
    print(
        f"First / last:   "
        f"{selection['base_first_frame']} / "
        f"{selection['base_last_frame']}"
    )

    print()

    for camera in plan["cameras"]:
        print(
            f"Camera {camera['camera_index']} "
            f"({camera['camera_name']}): "
            f"{camera['selected_count']} frames, "
            f"offset={camera['offset']}"
        )

    print()
    print(
        "Maximum images if both cameras are extracted:",
        plan["total_images_if_both_cameras_extracted"],
    )
    print("No images were extracted.")
    print()
    print(f"Plan: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
