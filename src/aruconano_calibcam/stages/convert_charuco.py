"""Stage 3B: convert raw ArUco Nano markers to CalibCam ChArUco detections."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from calibcam import helper
from calibcamlib import Board, Detections

from aruconano_calibcam.config import load_configuration


class CharucoConversionError(RuntimeError):
    """Raised when ChArUco conversion or validation fails."""


class NoAliasSafeDumper(yaml.SafeDumper):
    """Safe YAML dumper that never emits anchors or aliases."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


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
    """Load a JSON object."""

    if not path.is_file():
        raise CharucoConversionError(
            f"JSON file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise CharucoConversionError(
            f"JSON root is not an object: {path}"
        )

    return value


def frame_index(filename: str) -> int:
    """Extract the numeric source-frame index from a filename."""

    match = re.search(
        r"(\d+)",
        Path(filename).stem,
    )

    if not match:
        raise CharucoConversionError(
            f"Cannot extract frame index from: {filename}"
        )

    return int(match.group(1))


def load_marker_csv(
    path: Path,
) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Load raw ArUco Nano marker corners grouped by frame."""

    if not path.is_file():
        raise CharucoConversionError(
            f"Raw marker CSV does not exist: {path}"
        )

    grouped: dict[
        str,
        list[tuple[int, np.ndarray]],
    ] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required_columns = {
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
        }

        if (
            reader.fieldnames is None
            or not required_columns.issubset(
                reader.fieldnames
            )
        ):
            raise CharucoConversionError(
                "Raw marker CSV does not contain the "
                "required columns."
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            filename = row["frame"]

            try:
                marker_id = int(row["marker_id"])

                corners = np.asarray(
                    [
                        [
                            float(row["corner0_x"]),
                            float(row["corner0_y"]),
                        ],
                        [
                            float(row["corner1_x"]),
                            float(row["corner1_y"]),
                        ],
                        [
                            float(row["corner2_x"]),
                            float(row["corner2_y"]),
                        ],
                        [
                            float(row["corner3_x"]),
                            float(row["corner3_y"]),
                        ],
                    ],
                    dtype=np.float32,
                )
            except ValueError as error:
                raise CharucoConversionError(
                    "Invalid raw marker value at CSV row "
                    f"{row_number}."
                ) from error

            if not np.all(np.isfinite(corners)):
                raise CharucoConversionError(
                    "Non-finite raw marker corner at CSV row "
                    f"{row_number}."
                )

            grouped.setdefault(
                filename,
                [],
            ).append(
                (
                    marker_id,
                    corners,
                )
            )

    return grouped


def compare_detection_files(
    candidate_path: Path,
    reference_path: Path,
    tolerance_px: float,
) -> dict[str, Any]:
    """Compare a new detection file with a known-good reference."""

    candidate = Detections.from_file(
        candidate_path
    ).to_array()

    reference = Detections.from_file(
        reference_path
    ).to_array()

    structural_checks = {
        "marker_ids_match": np.array_equal(
            candidate["marker_ids"],
            reference["marker_ids"],
        ),
        "detection_idxs_match": np.array_equal(
            candidate["detection_idxs"],
            reference["detection_idxs"],
        ),
        "frame_idxs_match": np.array_equal(
            candidate["frame_idxs"],
            reference["frame_idxs"],
        ),
        "marker_coords_shape_match": (
            candidate["marker_coords"].shape
            == reference["marker_coords"].shape
        ),
    }

    maximum_difference_px: float | None = None
    finite_masks_match = False
    coordinates_within_tolerance = False

    if structural_checks["marker_coords_shape_match"]:
        candidate_coords = np.asarray(
            candidate["marker_coords"],
            dtype=float,
        )

        reference_coords = np.asarray(
            reference["marker_coords"],
            dtype=float,
        )

        candidate_finite = np.isfinite(
            candidate_coords
        )

        reference_finite = np.isfinite(
            reference_coords
        )

        finite_masks_match = np.array_equal(
            candidate_finite,
            reference_finite,
        )

        if finite_masks_match:
            differences = np.abs(
                candidate_coords[candidate_finite]
                - reference_coords[reference_finite]
            )

            maximum_difference_px = (
                float(np.max(differences))
                if differences.size
                else 0.0
            )

            coordinates_within_tolerance = (
                maximum_difference_px
                <= tolerance_px
            )

    checks = {
        **structural_checks,
        "finite_masks_match": finite_masks_match,
        "coordinates_within_tolerance": (
            coordinates_within_tolerance
        ),
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "tolerance_px": tolerance_px,
        "maximum_coordinate_difference_px": (
            maximum_difference_px
        ),
        "reference_file": str(
            reference_path.resolve()
        ),
        "reference_sha256": sha256(
            reference_path
        ),
    }


def convert_to_charuco(
    project_root: Path,
    configuration_path: Path,
    extraction_manifest_path: Path,
    marker_manifest_path: Path,
    camera_name: str,
    output_path: Path,
    reference_path: Path | None = None,
    regression_tolerance_px: float = 0.1,
) -> dict[str, Any]:
    """Convert one camera's raw marker CSV to CalibCam YAML."""

    project_root = project_root.resolve()
    configuration_path = configuration_path.resolve()
    extraction_manifest_path = (
        extraction_manifest_path.resolve()
    )
    marker_manifest_path = (
        marker_manifest_path.resolve()
    )
    output_path = output_path.resolve()

    temporary_path = output_path.with_name(
        f".{output_path.stem}.temporary"
        f"{output_path.suffix}"
    )

    if output_path.exists():
        raise CharucoConversionError(
            f"Output already exists: {output_path}"
        )

    if temporary_path.exists():
        raise CharucoConversionError(
            f"Temporary output already exists: "
            f"{temporary_path}"
        )

    configuration = load_configuration(
        configuration_path
    )

    extraction = load_json(
        extraction_manifest_path
    )

    marker_report = load_json(
        marker_manifest_path
    )

    dataset_id = configuration["dataset"]["id"]

    if extraction.get("stage") != "frame_extraction":
        raise CharucoConversionError(
            "Unexpected extraction manifest stage."
        )

    if marker_report.get("stage") != "marker_detection":
        raise CharucoConversionError(
            "Unexpected marker manifest stage."
        )

    if not extraction.get("valid", False):
        raise CharucoConversionError(
            "Extraction manifest is not valid."
        )

    if not marker_report.get("valid", False):
        raise CharucoConversionError(
            "Marker-detection manifest is not valid."
        )

    if (
        extraction["dataset_id"] != dataset_id
        or marker_report["dataset_id"] != dataset_id
    ):
        raise CharucoConversionError(
            "Dataset IDs do not match."
        )

    if (
        extraction["camera"]["name"] != camera_name
        or marker_report["camera"]["name"]
        != camera_name
    ):
        raise CharucoConversionError(
            "Camera names do not match."
        )

    configured_cameras = [
        camera
        for camera in configuration["cameras"]
        if camera["name"] == camera_name
    ]

    if len(configured_cameras) != 1:
        raise CharucoConversionError(
            f"Could not resolve camera: {camera_name}"
        )

    camera = configured_cameras[0]

    frame_directory = Path(
        extraction["output_directory"]
    ).resolve()

    marker_csv = Path(
        marker_report["outputs"]["detections_csv"]
    ).resolve()

    if not frame_directory.is_dir():
        raise CharucoConversionError(
            f"Frame directory is missing: "
            f"{frame_directory}"
        )

    board_path = (
        project_root
        / configuration["board"]["canonical_file"]
    ).resolve()

    if not board_path.is_file():
        raise CharucoConversionError(
            f"Canonical board is missing: "
            f"{board_path}"
        )

    board = Board.from_file(board_path)
    cv_board = board.get_cv2_board()

    legacy_pattern = bool(
        configuration["board"]["legacy_pattern"]
    )

    try:
        cv_board.setLegacyPattern(
            legacy_pattern
        )
    except AttributeError:
        pass

    charuco_detector = (
        cv2.aruco.CharucoDetector(cv_board)
    )

    board_parameters = board.get_board_params()

    board_width = int(
        board_parameters["boardWidth"]
    )

    board_ids = np.asarray(
        board.get_board_ids()
    ).reshape(-1)

    if board_ids.size == 0:
        raise CharucoConversionError(
            "The configured board contains no marker IDs."
        )

    valid_marker_ids = {
        int(value)
        for value in board_ids
    }

    board_start_id = int(board_ids[0])

    number_of_charuco_points = len(
        board.get_board_points()
    )

    minimum_points = int(
        configuration["detection"][
            "minimum_charuco_corners"
        ]
    )

    grouped = load_marker_csv(marker_csv)

    extracted_filenames = {
        item["filename"]
        for item in extraction["files"]
    }

    unknown_csv_frames = sorted(
        set(grouped) - extracted_filenames
    )

    if unknown_csv_frames:
        raise CharucoConversionError(
            "Raw marker CSV references unknown extracted "
            f"frames: {unknown_csv_frames[:10]}"
        )

    accepted: list[
        tuple[int, np.ndarray, np.ndarray]
    ] = []

    frames_with_nano = 0
    frames_with_charuco = 0
    fallback_interpolations = 0
    rejected_nonfinite = 0
    rejected_degenerate = 0

    for filename in sorted(
        grouped,
        key=frame_index,
    ):
        image_path = frame_directory / filename

        gray = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if gray is None:
            raise CharucoConversionError(
                f"Could not read extracted frame: "
                f"{image_path}"
            )

        items = [
            item
            for item in grouped[filename]
            if item[0] in valid_marker_ids
        ]

        if not items:
            continue

        frames_with_nano += 1

        marker_ids = np.asarray(
            [
                item[0]
                for item in items
            ],
            dtype=np.int32,
        ).reshape(-1, 1)

        marker_corners = [
            item[1].reshape(4, 2)
            for item in items
        ]

        try:
            (
                charuco_corners,
                charuco_ids,
                _,
                _,
            ) = charuco_detector.detectBoard(
                gray,
                None,
                None,
                marker_corners,
                marker_ids,
            )
        except Exception:
            fallback_interpolations += 1

            (
                _,
                charuco_corners,
                charuco_ids,
            ) = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                cv_board,
            )

        if (
            charuco_corners is None
            or charuco_ids is None
        ):
            continue

        ids_local = np.asarray(
            charuco_ids,
            dtype=int,
        ).reshape(-1)

        points = np.asarray(
            charuco_corners,
            dtype=float,
        ).reshape(-1, 2)

        if ids_local.size != points.shape[0]:
            raise CharucoConversionError(
                "ChArUco ID and coordinate counts differ "
                f"for {filename}."
            )

        if ids_local.size == 0:
            continue

        frames_with_charuco += 1

        ids_calibcam = (
            ids_local + board_start_id
        )

        if not np.all(np.isfinite(points)):
            rejected_nonfinite += 1
            continue

        if not helper.check_detections_nondegenerate(
            board_width,
            ids_calibcam,
            minimum_points=minimum_points,
        ):
            rejected_degenerate += 1
            continue

        accepted.append(
            (
                frame_index(filename),
                ids_calibcam,
                points,
            )
        )

    if not accepted:
        raise CharucoConversionError(
            f"No valid ChArUco detections for "
            f"{camera_name}."
        )

    detection_indices = [
        int(index)
        for index, _, _ in accepted
    ]

    if detection_indices != sorted(
        set(detection_indices)
    ):
        raise CharucoConversionError(
            "Accepted frame indices are not sorted "
            "and unique."
        )

    active_ids = sorted(
        {
            int(marker_id)
            for _, ids, _ in accepted
            for marker_id in ids
        }
    )

    id_to_column = {
        marker_id: column
        for column, marker_id
        in enumerate(active_ids)
    }

    marker_coordinates: list[
        list[list[float]]
    ] = []

    corner_counts: list[int] = []

    for _, ids, points in accepted:
        coordinates = np.full(
            (len(active_ids), 2),
            np.nan,
            dtype=float,
        )

        for marker_id, point in zip(
            ids,
            points,
        ):
            coordinates[
                id_to_column[int(marker_id)]
            ] = point

        marker_coordinates.append(
            coordinates.tolist()
        )

        corner_counts.append(
            int(len(ids))
        )

    # Independent list objects deliberately prevent YAML anchors.
    payload = {
        "version": 1,
        "storage_method": "calibcam",
        "marker_ids": list(active_ids),
        "detection_idxs": list(
            detection_indices
        ),
        "frame_idxs": [
            list(detection_indices)
        ],
        "marker_coords": [
            marker_coordinates
        ],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.dump(
            payload,
            handle,
            Dumper=NoAliasSafeDumper,
            sort_keys=False,
            default_flow_style=False,
        )

    raw_yaml = temporary_path.read_text(
        encoding="utf-8"
    )

    if "&id" in raw_yaml or "*id" in raw_yaml:
        raise CharucoConversionError(
            "Generated YAML unexpectedly contains "
            "an anchor or alias."
        )

    with temporary_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        reloaded_payload = yaml.safe_load(handle)

    required_keys = [
        "version",
        "storage_method",
        "marker_ids",
        "detection_idxs",
        "frame_idxs",
        "marker_coords",
    ]

    if list(reloaded_payload) != required_keys:
        raise CharucoConversionError(
            "Generated YAML keys or key order differ "
            "from the required contract."
        )

    loaded = Detections.from_file(
        temporary_path
    ).to_array()

    expected_shape = (
        1,
        len(accepted),
        len(active_ids),
        2,
    )

    if (
        loaded["marker_coords"].shape
        != expected_shape
    ):
        raise CharucoConversionError(
            "CalibCam-loaded marker coordinate shape "
            f"is {loaded['marker_coords'].shape}; "
            f"expected {expected_shape}."
        )

    if not np.array_equal(
        loaded["marker_ids"],
        np.asarray(active_ids),
    ):
        raise CharucoConversionError(
            "CalibCam-loaded marker IDs differ."
        )

    if not np.array_equal(
        loaded["detection_idxs"],
        np.asarray(detection_indices),
    ):
        raise CharucoConversionError(
            "CalibCam-loaded detection indices differ."
        )

    if not np.array_equal(
        loaded["frame_idxs"],
        np.asarray([detection_indices]),
    ):
        raise CharucoConversionError(
            "CalibCam-loaded frame indices differ."
        )

    regression: dict[str, Any] | None = None

    if reference_path is not None:
        reference_path = reference_path.resolve()

        if not reference_path.is_file():
            raise CharucoConversionError(
                f"Reference YAML does not exist: "
                f"{reference_path}"
            )

        regression = compare_detection_files(
            candidate_path=temporary_path,
            reference_path=reference_path,
            tolerance_px=regression_tolerance_px,
        )

        if not regression["passed"]:
            raise CharucoConversionError(
                "Generated ChArUco detections do not "
                "match the known-good reference. "
                f"Regression details: {regression}"
            )

    temporary_path.replace(output_path)

    return {
        "schema_version": 1,
        "stage": "charuco_conversion",
        "dataset_id": dataset_id,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "valid": True,
        "configuration": str(
            configuration_path
        ),
        "extraction_manifest": str(
            extraction_manifest_path
        ),
        "marker_detection_manifest": str(
            marker_manifest_path
        ),
        "camera": {
            "index": int(camera["index"]),
            "name": camera_name,
        },
        "board": {
            "file": str(board_path),
            "sha256": sha256(board_path),
            "legacy_pattern": legacy_pattern,
            "board_width": board_width,
            "board_start_id": board_start_id,
            "board_marker_ids": sorted(
                valid_marker_ids
            ),
            "number_of_charuco_points": (
                number_of_charuco_points
            ),
        },
        "conversion": {
            "minimum_points": minimum_points,
            "frames_with_nano_markers": (
                frames_with_nano
            ),
            "frames_with_charuco": (
                frames_with_charuco
            ),
            "valid_calibcam_frames": len(
                accepted
            ),
            "fallback_interpolations": (
                fallback_interpolations
            ),
            "rejected_nonfinite": (
                rejected_nonfinite
            ),
            "rejected_degenerate": (
                rejected_degenerate
            ),
            "active_charuco_ids": active_ids,
            "active_charuco_id_count": len(
                active_ids
            ),
            "mean_corners_per_frame": float(
                np.mean(corner_counts)
            ),
            "median_corners_per_frame": float(
                np.median(corner_counts)
            ),
            "minimum_corners_per_frame": int(
                np.min(corner_counts)
            ),
            "maximum_corners_per_frame": int(
                np.max(corner_counts)
            ),
            "first_frame_index": (
                detection_indices[0]
            ),
            "last_frame_index": (
                detection_indices[-1]
            ),
            "detection_indices": (
                detection_indices
            ),
        },
        "output": {
            "yaml": str(output_path),
            "sha256": sha256(output_path),
            "size_bytes": output_path.stat().st_size,
            "contains_yaml_anchor": False,
            "calibcam_marker_coords_shape": list(
                expected_shape
            ),
        },
        "regression": regression,
    }
