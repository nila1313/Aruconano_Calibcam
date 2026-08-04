"""Stage 3A: run and validate native ArUco Nano marker detection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from calibcamlib import Board

from aruconano_calibcam.config import load_configuration


class MarkerDetectionError(RuntimeError):
    """Raised when native marker detection fails validation."""


DETECTION_COLUMNS = [
    "frame",
    "marker_id",
    "corner0_x",
    "corner0_y",
    "corner1_x",
    "corner1_y",
    "corner2_x",
    "corner2_y",
    "corner3_x",
    "corner3_y",
    "center_x",
    "center_y",
]

SUMMARY_COLUMNS = [
    "frame",
    "num_markers",
]

COORDINATE_COLUMNS = DETECTION_COLUMNS[2:]


def sha256(path: Path) -> str:
    """Calculate a file SHA-256 checksum."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON dictionary."""

    if not path.is_file():
        raise MarkerDetectionError(
            f"JSON file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise MarkerDetectionError(
            f"JSON root must be an object: {path}"
        )

    return value


def load_summary(
    path: Path,
) -> tuple[list[str], dict[str, int]]:
    """Load and validate frame_summary.csv."""

    if not path.is_file():
        raise MarkerDetectionError(
            f"Summary CSV is missing: {path}"
        )

    ordered_frames: list[str] = []
    summary: dict[str, int] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != SUMMARY_COLUMNS:
            raise MarkerDetectionError(
                "Unexpected summary CSV columns: "
                f"{reader.fieldnames}"
            )

        for row in reader:
            frame = row["frame"]

            if frame in summary:
                raise MarkerDetectionError(
                    f"Duplicate summary frame: {frame}"
                )

            try:
                marker_count = int(row["num_markers"])
            except ValueError as error:
                raise MarkerDetectionError(
                    f"Invalid marker count for {frame}."
                ) from error

            if marker_count < 0:
                raise MarkerDetectionError(
                    f"Negative marker count for {frame}."
                )

            ordered_frames.append(frame)
            summary[frame] = marker_count

    return ordered_frames, summary


def load_detections(
    path: Path,
    expected_frames: set[str],
    dictionary_capacity: int | None,
) -> list[dict[str, Any]]:
    """Load and validate detections.csv."""

    if not path.is_file():
        raise MarkerDetectionError(
            f"Detections CSV is missing: {path}"
        )

    detections: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != DETECTION_COLUMNS:
            raise MarkerDetectionError(
                "Unexpected detection CSV columns: "
                f"{reader.fieldnames}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            frame = row["frame"]

            if frame not in expected_frames:
                raise MarkerDetectionError(
                    "Detection references an unknown frame "
                    f"at CSV row {row_number}: {frame}"
                )

            try:
                marker_id = int(row["marker_id"])
            except ValueError as error:
                raise MarkerDetectionError(
                    f"Invalid marker ID at CSV row {row_number}."
                ) from error

            if marker_id < 0:
                raise MarkerDetectionError(
                    f"Negative marker ID at CSV row {row_number}."
                )

            if (
                dictionary_capacity is not None
                and marker_id >= dictionary_capacity
            ):
                raise MarkerDetectionError(
                    f"Marker ID {marker_id} exceeds dictionary "
                    f"capacity {dictionary_capacity}."
                )

            key = (frame, marker_id)

            if key in seen_keys:
                raise MarkerDetectionError(
                    "Duplicate marker ID in one frame: "
                    f"{frame}, marker {marker_id}"
                )

            seen_keys.add(key)

            coordinates: dict[str, float] = {}

            for field in COORDINATE_COLUMNS:
                try:
                    value = float(row[field])
                except ValueError as error:
                    raise MarkerDetectionError(
                        f"Invalid coordinate {field} "
                        f"at CSV row {row_number}."
                    ) from error

                if not math.isfinite(value):
                    raise MarkerDetectionError(
                        f"Non-finite coordinate {field} "
                        f"at CSV row {row_number}."
                    )

                coordinates[field] = value

            expected_center_x = sum(
                coordinates[f"corner{index}_x"]
                for index in range(4)
            ) / 4.0

            expected_center_y = sum(
                coordinates[f"corner{index}_y"]
                for index in range(4)
            ) / 4.0

            center_error = max(
                abs(
                    coordinates["center_x"]
                    - expected_center_x
                ),
                abs(
                    coordinates["center_y"]
                    - expected_center_y
                ),
            )

            # CSV decimal formatting can introduce a tiny difference.
            if center_error > 0.01:
                raise MarkerDetectionError(
                    "Marker center does not match the mean of "
                    f"its corners at CSV row {row_number}: "
                    f"{center_error:.6f} px"
                )

            detections.append(
                {
                    "frame": frame,
                    "marker_id": marker_id,
                    "coordinates": coordinates,
                }
            )

    return detections


def dictionary_capacity(
    dictionary_name: str,
) -> int | None:
    """Infer dictionary capacity from names such as DICT_6X6_250."""

    final_token = dictionary_name.rsplit("_", 1)[-1]

    try:
        return int(final_token)
    except ValueError:
        return None


def board_marker_ids(
    project_root: Path,
    configuration: dict[str, Any],
) -> list[int]:
    """Read marker IDs belonging to the configured ChArUco board."""

    board_path = (
        project_root
        / configuration["board"]["canonical_file"]
    )

    board = Board.from_file(
        str(board_path.with_suffix(""))
    )

    ids = np.asarray(
        board.get_board_ids()
    ).reshape(-1)

    return sorted(
        {int(marker_id) for marker_id in ids}
    )


def run_marker_detection(
    project_root: Path,
    configuration_path: Path,
    extraction_manifest_path: Path,
    camera_name: str,
    detector_binary: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Run ArUco Nano and validate its raw CSV outputs."""

    project_root = project_root.resolve()
    configuration_path = configuration_path.resolve()
    extraction_manifest_path = (
        extraction_manifest_path.resolve()
    )
    detector_binary = detector_binary.resolve()
    output_directory = output_directory.resolve()

    temporary_output = output_directory.with_name(
        f"{output_directory.name}.temporary"
    )

    configuration = load_configuration(
        configuration_path
    )

    extraction = load_json(
        extraction_manifest_path
    )

    if extraction.get("stage") != "frame_extraction":
        raise MarkerDetectionError(
            "Unexpected extraction manifest stage."
        )

    if not extraction.get("valid", False):
        raise MarkerDetectionError(
            "Extraction manifest is not valid."
        )

    if (
        extraction["dataset_id"]
        != configuration["dataset"]["id"]
    ):
        raise MarkerDetectionError(
            "Configuration and extraction dataset IDs differ."
        )

    if extraction["camera"]["name"] != camera_name:
        raise MarkerDetectionError(
            "Extraction camera does not match requested camera."
        )

    configured_cameras = [
        camera
        for camera in configuration["cameras"]
        if camera["name"] == camera_name
    ]

    if len(configured_cameras) != 1:
        raise MarkerDetectionError(
            f"Could not resolve camera: {camera_name}"
        )

    camera = configured_cameras[0]

    if not detector_binary.is_file():
        raise MarkerDetectionError(
            f"Detector binary is missing: {detector_binary}"
        )

    if not os.access(detector_binary, os.X_OK):
        raise MarkerDetectionError(
            f"Detector is not executable: {detector_binary}"
        )

    input_directory = Path(
        extraction["output_directory"]
    ).resolve()

    if not input_directory.is_dir():
        raise MarkerDetectionError(
            f"Extracted-frame directory is missing: "
            f"{input_directory}"
        )

    if output_directory.exists():
        raise MarkerDetectionError(
            f"Output directory already exists: "
            f"{output_directory}"
        )

    if temporary_output.exists():
        raise MarkerDetectionError(
            f"Temporary output already exists: "
            f"{temporary_output}"
        )

    expected_filenames = [
        item["filename"]
        for item in extraction["files"]
    ]

    filesystem_filenames = sorted(
        path.name
        for path in input_directory.glob("frame_*.png")
        if path.is_file()
    )

    if filesystem_filenames != expected_filenames:
        raise MarkerDetectionError(
            "Extracted PNG files do not match the "
            "extraction manifest."
        )

    detector_backend = configuration["detection"]["backend"]

    if detector_backend != "aruco_nano":
        raise MarkerDetectionError(
            f"Unsupported detector backend: {detector_backend}"
        )

    dictionary_id = int(
        configuration["board"]["dictionary_id"]
    )

    dictionary_name = str(
        configuration["board"]["dictionary_name"]
    )

    capacity = dictionary_capacity(dictionary_name)

    command = [
        str(detector_binary),
        str(input_directory),
        str(temporary_output),
        str(dictionary_id),
    ]

    print("Running native detector:")
    print(" ".join(command))
    print()

    started = time.perf_counter()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []

    if process.stdout is None:
        raise MarkerDetectionError(
            "Could not capture detector output."
        )

    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    return_code = process.wait()
    duration_seconds = time.perf_counter() - started

    if return_code != 0:
        raise MarkerDetectionError(
            "Native detector failed with return code "
            f"{return_code}. Partial output remains at "
            f"{temporary_output}"
        )

    summary_path = (
        temporary_output / "frame_summary.csv"
    )

    detections_path = (
        temporary_output / "detections.csv"
    )

    ordered_summary_frames, summary = load_summary(
        summary_path
    )

    if ordered_summary_frames != expected_filenames:
        raise MarkerDetectionError(
            "Detector frame summary does not exactly match "
            "the extraction manifest."
        )

    expected_frame_set = set(expected_filenames)

    detections = load_detections(
        detections_path,
        expected_frame_set,
        capacity,
    )

    observed_counts = Counter(
        detection["frame"]
        for detection in detections
    )

    count_mismatches = {
        frame: {
            "summary": summary[frame],
            "detection_rows": observed_counts.get(
                frame,
                0,
            ),
        }
        for frame in expected_filenames
        if summary[frame]
        != observed_counts.get(frame, 0)
    }

    if count_mismatches:
        raise MarkerDetectionError(
            "Frame summary counts do not match detection "
            f"rows: {count_mismatches}"
        )

    observed_marker_ids = sorted(
        {
            int(detection["marker_id"])
            for detection in detections
        }
    )

    expected_board_ids = board_marker_ids(
        project_root,
        configuration,
    )

    unexpected_marker_ids = sorted(
        set(observed_marker_ids)
        - set(expected_board_ids)
    )

    missing_board_ids = sorted(
        set(expected_board_ids)
        - set(observed_marker_ids)
    )

    zero_marker_frames = [
        frame
        for frame in expected_filenames
        if summary[frame] == 0
    ]

    frames_with_markers = [
        frame
        for frame in expected_filenames
        if summary[frame] > 0
    ]

    marker_counts = list(summary.values())

    output_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output.replace(output_directory)

    final_summary_path = (
        output_directory / "frame_summary.csv"
    )

    final_detections_path = (
        output_directory / "detections.csv"
    )

    warnings: list[str] = []

    if unexpected_marker_ids:
        warnings.append(
            "Detected marker IDs not belonging to the "
            f"configured board: {unexpected_marker_ids}"
        )

    if zero_marker_frames:
        warnings.append(
            f"{len(zero_marker_frames)} selected frame(s) "
            "contained no detected markers."
        )

    return {
        "schema_version": 1,
        "stage": "marker_detection",
        "dataset_id": configuration["dataset"]["id"],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "valid": True,
        "configuration": str(configuration_path),
        "extraction_manifest": str(
            extraction_manifest_path
        ),
        "camera": {
            "index": int(camera["index"]),
            "name": camera_name,
        },
        "detector": {
            "backend": detector_backend,
            "binary": str(detector_binary),
            "binary_sha256": sha256(detector_binary),
            "dictionary_id": dictionary_id,
            "dictionary_name": dictionary_name,
            "dictionary_capacity": capacity,
            "opencl": False,
            "opencv_threads": 1,
            "return_code": return_code,
            "duration_seconds": round(
                duration_seconds,
                6,
            ),
            "console_output": "".join(output_lines),
        },
        "outputs": {
            "directory": str(output_directory),
            "detections_csv": str(
                final_detections_path
            ),
            "detections_sha256": sha256(
                final_detections_path
            ),
            "summary_csv": str(final_summary_path),
            "summary_sha256": sha256(
                final_summary_path
            ),
        },
        "statistics": {
            "frames_processed": len(expected_filenames),
            "frames_with_markers": len(
                frames_with_markers
            ),
            "frames_without_markers": len(
                zero_marker_frames
            ),
            "zero_marker_frames": zero_marker_frames,
            "total_marker_detections": len(detections),
            "minimum_markers_per_frame": min(
                marker_counts
            ),
            "maximum_markers_per_frame": max(
                marker_counts
            ),
            "observed_marker_ids": observed_marker_ids,
            "expected_board_marker_ids": (
                expected_board_ids
            ),
            "unexpected_marker_ids": (
                unexpected_marker_ids
            ),
            "missing_board_marker_ids": (
                missing_board_ids
            ),
        },
        "warnings": warnings,
    }
