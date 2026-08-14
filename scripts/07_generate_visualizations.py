#!/usr/bin/env python3
"""Generate annotated ArUco and ChArUco images from validated outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import cv2
import numpy as np
import yaml
from calibcamlib import Board

from aruconano_calibcam.stages.convert_charuco import (
    frame_index,
    load_marker_csv,
)


class VisualizationError(RuntimeError):
    """Raised when visualization generation cannot be validated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise VisualizationError(
            f"JSON file is not an object: {path}"
        )

    return value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise VisualizationError(
            f"YAML file is not a dictionary: {path}"
        )

    return value


def prepare_temporary_directory(
    destination: Path,
) -> Path:
    temporary = destination.with_name(
        destination.name + ".temporary"
    )

    if temporary.exists():
        raise VisualizationError(
            f"Temporary directory already exists: {temporary}"
        )

    if destination.exists():
        if not destination.is_dir():
            raise VisualizationError(
                f"Destination is not a directory: {destination}"
            )

        if any(destination.iterdir()):
            raise VisualizationError(
                f"Destination is not empty: {destination}"
            )

    temporary.mkdir(parents=True, exist_ok=False)

    return temporary


def write_png(path: Path, image: np.ndarray) -> None:
    written = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )

    if not written or not path.is_file():
        raise VisualizationError(
            f"Could not write annotated image: {path}"
        )


def add_status_text(
    image: np.ndarray,
    text: str,
) -> None:
    cv2.putText(
        image,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def run_visualization(
    project_root: Path,
    configuration_path: Path,
    extraction_manifest_path: Path,
    marker_manifest_path: Path,
    charuco_detection_path: Path,
    camera_name: str,
    aruco_output_directory: Path,
    charuco_output_directory: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    configuration_path = configuration_path.resolve()
    extraction_manifest_path = (
        extraction_manifest_path.resolve()
    )
    marker_manifest_path = marker_manifest_path.resolve()
    charuco_detection_path = (
        charuco_detection_path.resolve()
    )
    aruco_output_directory = (
        aruco_output_directory.resolve()
    )
    charuco_output_directory = (
        charuco_output_directory.resolve()
    )
    manifest_path = manifest_path.resolve()

    for path in (
        configuration_path,
        extraction_manifest_path,
        marker_manifest_path,
        charuco_detection_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    configuration = load_yaml(configuration_path)
    extraction = load_json(extraction_manifest_path)
    marker_manifest = load_json(marker_manifest_path)
    charuco_payload = load_yaml(charuco_detection_path)

    if not configuration["detection"].get(
        "write_annotated_frames",
        False,
    ):
        raise VisualizationError(
            "write_annotated_frames is not enabled."
        )

    if extraction.get("stage") != "frame_extraction":
        raise VisualizationError(
            "Unexpected extraction manifest stage."
        )

    if marker_manifest.get("stage") != "marker_detection":
        raise VisualizationError(
            "Unexpected marker manifest stage."
        )

    if extraction["camera"]["name"] != camera_name:
        raise VisualizationError(
            "Extraction camera does not match."
        )

    if marker_manifest["camera"]["name"] != camera_name:
        raise VisualizationError(
            "Marker camera does not match."
        )

    frame_directory = Path(
        extraction["output_directory"]
    ).resolve()

    marker_csv = Path(
        marker_manifest["outputs"]["detections_csv"]
    ).resolve()

    board_path = (
        project_root
        / configuration["board"]["canonical_file"]
    ).resolve()

    for path in (
        frame_directory,
        marker_csv,
        board_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    board = Board.from_file(board_path)
    board_ids = np.asarray(
        board.get_board_ids(),
        dtype=int,
    ).reshape(-1)

    if board_ids.size == 0:
        raise VisualizationError(
            "Configured board contains no ArUco IDs."
        )

    board_start_id = int(board_ids[0])
    valid_board_ids = {
        int(value)
        for value in board_ids
    }

    grouped_markers = load_marker_csv(marker_csv)

    active_ids = np.asarray(
        charuco_payload["marker_ids"],
        dtype=int,
    ).reshape(-1)

    detection_indices = [
        int(value)
        for value in charuco_payload["detection_idxs"]
    ]

    marker_coordinates = np.asarray(
        charuco_payload["marker_coords"],
        dtype=float,
    )

    expected_shape = (
        1,
        len(detection_indices),
        len(active_ids),
        2,
    )

    if marker_coordinates.shape != expected_shape:
        raise VisualizationError(
            "Unexpected ChArUco coordinate shape: "
            f"{marker_coordinates.shape}; "
            f"expected {expected_shape}"
        )

    charuco_by_frame: dict[
        int,
        tuple[np.ndarray, np.ndarray],
    ] = {}

    for row, physical_index in enumerate(
        detection_indices
    ):
        coordinates = marker_coordinates[0, row]
        finite = np.all(np.isfinite(coordinates), axis=1)

        charuco_by_frame[physical_index] = (
            active_ids[finite],
            coordinates[finite],
        )

    expected_files = sorted(
        extraction["files"],
        key=lambda item: frame_index(item["filename"]),
    )

    expected_names = [
        item["filename"]
        for item in expected_files
    ]

    actual_names = sorted(
        path.name
        for path in frame_directory.glob("frame_*.png")
        if path.is_file()
    )

    if actual_names != sorted(expected_names):
        raise VisualizationError(
            "Extracted files do not match the manifest."
        )

    aruco_temporary = prepare_temporary_directory(
        aruco_output_directory
    )

    charuco_temporary = prepare_temporary_directory(
        charuco_output_directory
    )

    aruco_with_detections = 0
    charuco_with_detections = 0

    try:
        for item in expected_files:
            filename = item["filename"]
            source = frame_directory / filename

            image = cv2.imread(
                str(source),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                raise VisualizationError(
                    f"Could not read frame: {source}"
                )

            index = frame_index(filename)
            stem = Path(filename).stem

            aruco_image = image.copy()

            marker_items = [
                marker
                for marker in grouped_markers.get(
                    filename,
                    [],
                )
                if int(marker[0]) in valid_board_ids
            ]

            if marker_items:
                corners = [
                    np.asarray(
                        marker[1],
                        dtype=np.float32,
                    ).reshape(1, 4, 2)
                    for marker in marker_items
                ]

                ids = np.asarray(
                    [
                        int(marker[0])
                        for marker in marker_items
                    ],
                    dtype=np.int32,
                ).reshape(-1, 1)

                cv2.aruco.drawDetectedMarkers(
                    aruco_image,
                    corners,
                    ids,
                )

                aruco_with_detections += 1
            else:
                add_status_text(
                    aruco_image,
                    "No board ArUco markers detected",
                )

            aruco_target = (
                aruco_temporary
                / f"{stem}_aruco.png"
            )

            write_png(
                aruco_target,
                aruco_image,
            )

            charuco_image = image.copy()
            charuco_entry = charuco_by_frame.get(index)

            if charuco_entry is not None:
                ids_calibcam, points = charuco_entry

                if len(ids_calibcam) > 0:
                    ids_local = (
                        ids_calibcam - board_start_id
                    ).astype(np.int32).reshape(-1, 1)

                    corners = np.asarray(
                        points,
                        dtype=np.float32,
                    ).reshape(-1, 1, 2)

                    cv2.aruco.drawDetectedCornersCharuco(
                        charuco_image,
                        corners,
                        ids_local,
                    )

                    charuco_with_detections += 1
                else:
                    add_status_text(
                        charuco_image,
                        "No accepted ChArUco corners",
                    )
            else:
                add_status_text(
                    charuco_image,
                    "No accepted ChArUco corners",
                )

            charuco_target = (
                charuco_temporary
                / f"{stem}_charuco.png"
            )

            write_png(
                charuco_target,
                charuco_image,
            )

        aruco_written = sorted(
            aruco_temporary.glob("*.png")
        )

        charuco_written = sorted(
            charuco_temporary.glob("*.png")
        )

        if len(aruco_written) != len(expected_files):
            raise VisualizationError(
                "Unexpected ArUco visualization count."
            )

        if len(charuco_written) != len(expected_files):
            raise VisualizationError(
                "Unexpected ChArUco visualization count."
            )

        if aruco_output_directory.exists():
            aruco_output_directory.rmdir()

        if charuco_output_directory.exists():
            charuco_output_directory.rmdir()

        aruco_temporary.replace(
            aruco_output_directory
        )

        charuco_temporary.replace(
            charuco_output_directory
        )

    except Exception:
        if aruco_temporary.exists():
            shutil.rmtree(aruco_temporary)

        if charuco_temporary.exists():
            shutil.rmtree(charuco_temporary)

        raise

    result = {
        "schema_version": 1,
        "stage": "visualization_generation",
        "dataset_id": configuration["dataset"]["id"],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "valid": True,
        "camera": camera_name,
        "configuration": str(configuration_path),
        "extraction_manifest": str(
            extraction_manifest_path
        ),
        "marker_manifest": str(marker_manifest_path),
        "charuco_detection": str(
            charuco_detection_path
        ),
        "board": {
            "path": str(board_path),
            "sha256": sha256(board_path),
            "unit": configuration["board"]["unit"],
            "square_size_x_m": configuration[
                "board"
            ]["square_size_x_m"],
            "square_size_y_m": configuration[
                "board"
            ]["square_size_y_m"],
        },
        "aruco_visualizations": {
            "directory": str(
                aruco_output_directory
            ),
            "image_count": len(expected_files),
            "frames_with_detections":
                aruco_with_detections,
        },
        "charuco_visualizations": {
            "directory": str(
                charuco_output_directory
            ),
            "image_count": len(expected_files),
            "frames_with_accepted_corners":
                charuco_with_detections,
        },
    }

    if manifest_path.exists():
        raise VisualizationError(
            f"Manifest already exists: {manifest_path}"
        )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_manifest = manifest_path.with_suffix(
        manifest_path.suffix + ".temporary"
    )

    with temporary_manifest.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )
        handle.write("\n")

    temporary_manifest.replace(manifest_path)

    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate validated ArUco and ChArUco "
            "annotated frame images."
        )
    )

    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--extraction-manifest",
        required=True,
    )
    parser.add_argument(
        "--marker-manifest",
        required=True,
    )
    parser.add_argument(
        "--charuco-detection",
        required=True,
    )
    parser.add_argument(
        "--camera",
        required=True,
        choices=("left", "right"),
    )
    parser.add_argument(
        "--aruco-output-dir",
        required=True,
    )
    parser.add_argument(
        "--charuco-output-dir",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    result = run_visualization(
        project_root=project_root,
        configuration_path=Path(arguments.config),
        extraction_manifest_path=Path(
            arguments.extraction_manifest
        ),
        marker_manifest_path=Path(
            arguments.marker_manifest
        ),
        charuco_detection_path=Path(
            arguments.charuco_detection
        ),
        camera_name=arguments.camera,
        aruco_output_directory=Path(
            arguments.aruco_output_dir
        ),
        charuco_output_directory=Path(
            arguments.charuco_output_dir
        ),
        manifest_path=Path(arguments.manifest),
    )

    print()
    print("Visualization generation")
    print("========================")
    print("Camera:", result["camera"])
    print(
        "ArUco images:",
        result["aruco_visualizations"][
            "image_count"
        ],
    )
    print(
        "ChArUco images:",
        result["charuco_visualizations"][
            "image_count"
        ],
    )
    print(
        "ArUco frames with detections:",
        result["aruco_visualizations"][
            "frames_with_detections"
        ],
    )
    print(
        "ChArUco frames with accepted corners:",
        result["charuco_visualizations"][
            "frames_with_accepted_corners"
        ],
    )
    print()
    print("VISUALIZATION GENERATION: PASSED")


if __name__ == "__main__":
    main()
