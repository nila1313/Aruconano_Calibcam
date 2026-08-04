#!/usr/bin/env python3
"""Extract selected frames for one configured camera."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SOURCE_ROOT))

from aruconano_calibcam.stages.extract_frames import (  # noqa: E402
    extract_camera_frames,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract planned frames for one camera."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Dataset configuration YAML file.",
    )

    parser.add_argument(
        "--plan",
        required=True,
        type=Path,
        help="Generated frame-plan JSON file.",
    )

    parser.add_argument(
        "--camera",
        required=True,
        choices=("left", "right"),
        help="Camera to extract.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for extracted PNG images.",
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Destination extraction-manifest JSON file.",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    manifest = extract_camera_frames(
        project_root=PROJECT_ROOT,
        configuration_path=arguments.config,
        frame_plan_path=arguments.plan,
        camera_name=arguments.camera,
        output_directory=arguments.output_dir,
    )

    manifest_path = arguments.manifest.resolve()
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if manifest_path.exists():
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}"
        )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    size_mib = manifest["total_size_bytes"] / (1024**2)

    print()
    print("Frame extraction")
    print("================")
    print(f"Dataset:   {manifest['dataset_id']}")
    print(
        f"Camera:    "
        f"{manifest['camera']['index']} "
        f"({manifest['camera']['name']})"
    )
    print(
        f"Extracted: "
        f"{manifest['extracted_count']} / "
        f"{manifest['planned_count']}"
    )
    print(
        f"First/last: "
        f"{manifest['first_frame']} / "
        f"{manifest['last_frame']}"
    )
    print(f"PNG size:  {size_mib:.2f} MiB")
    print(f"Valid:     {manifest['valid']}")
    print()
    print(f"Frames:   {manifest['output_directory']}")
    print(f"Manifest: {manifest_path}")

    return 0 if manifest["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
