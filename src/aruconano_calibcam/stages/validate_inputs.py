"""Stage 1: validate pipeline inputs before processing."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from aruconano_calibcam.config import load_configuration


class InputValidationError(RuntimeError):
    """Raised when required pipeline inputs are invalid."""


def sha256(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    """Read the checked-out commit of a Git repository."""

    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def inspect_video(path: Path) -> dict[str, Any]:
    """Read video metadata and confirm several frames are decodable."""

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise InputValidationError(
            f"OpenCV could not open video: {path}"
        )

    frame_count = int(
        round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    decode_indices = sorted(
        {
            0,
            frame_count // 2,
            max(frame_count - 1, 0),
        }
    )

    decoded_frames: dict[str, bool] = {}

    for frame_index in decode_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()

        decoded_frames[str(frame_index)] = bool(
            success and frame is not None
        )

    capture.release()

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "decoded_frames": decoded_frames,
        "all_decode_tests_passed": all(
            decoded_frames.values()
        ),
    }


def validate_inputs(
    project_root: Path,
    configuration_path: Path,
) -> dict[str, Any]:
    """Validate all Pair 1 inputs and return a structured report."""

    project_root = project_root.resolve()
    configuration_path = configuration_path.resolve()

    configuration = load_configuration(configuration_path)

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    def add_check(
        name: str,
        passed: bool,
        details: Any,
    ) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "details": details,
            }
        )

        if not passed:
            errors.append(name)

    camera_reports: list[dict[str, Any]] = []

    for camera in configuration["cameras"]:
        video_path = project_root / camera["video"]
        exists = video_path.is_file()

        add_check(
            f"camera_{camera['index']}_video_exists",
            exists,
            str(video_path),
        )

        if not exists:
            continue

        actual_hash = sha256(video_path)
        expected_hash = camera["sha256"]

        add_check(
            f"camera_{camera['index']}_checksum",
            actual_hash == expected_hash,
            {
                "expected": expected_hash,
                "actual": actual_hash,
            },
        )

        metadata = inspect_video(video_path)

        metadata_match = all(
            [
                metadata["frame_count"]
                == camera["frame_count"],
                np.isclose(metadata["fps"], camera["fps"]),
                metadata["width"] == camera["width"],
                metadata["height"] == camera["height"],
            ]
        )

        add_check(
            f"camera_{camera['index']}_metadata",
            metadata_match,
            metadata,
        )

        add_check(
            f"camera_{camera['index']}_decode",
            metadata["all_decode_tests_passed"],
            metadata["decoded_frames"],
        )

        camera_reports.append(
            {
                "index": camera["index"],
                "name": camera["name"],
                "path": str(video_path),
                "metadata": metadata,
            }
        )

    if len(camera_reports) == 2:
        left = camera_reports[0]["metadata"]
        right = camera_reports[1]["metadata"]

        stereo_metadata_match = all(
            [
                left["frame_count"] == right["frame_count"],
                np.isclose(left["fps"], right["fps"]),
                left["width"] == right["width"],
                left["height"] == right["height"],
            ]
        )

        add_check(
            "stereo_video_metadata_match",
            stereo_metadata_match,
            {
                "left": left,
                "right": right,
            },
        )

    board_path = (
        project_root
        / configuration["board"]["canonical_file"]
    )

    board_exists = board_path.is_file()

    add_check(
        "canonical_board_exists",
        board_exists,
        str(board_path),
    )

    if board_exists:
        actual_board_hash = sha256(board_path)
        expected_board_hash = (
            configuration["board"]["canonical_sha256"]
        )

        add_check(
            "canonical_board_checksum",
            actual_board_hash == expected_board_hash,
            {
                "expected": expected_board_hash,
                "actual": actual_board_hash,
            },
        )

        board_parameters = np.load(
            board_path,
            allow_pickle=True,
        ).item()

        expected_board_values = {
            "boardWidth": configuration["board"]["width"],
            "boardHeight": configuration["board"]["height"],
            "dictionary_type": configuration["board"][
                "dictionary_id"
            ],
            "unit": configuration["board"]["unit"],
            "legacy": configuration["board"][
                "legacy_pattern"
            ],
        }

        board_values_match = all(
            board_parameters.get(key) == value
            for key, value in expected_board_values.items()
        )

        add_check(
            "canonical_board_parameters",
            board_values_match,
            {
                "expected": expected_board_values,
                "actual": board_parameters,
            },
        )

    versions_path = (
        project_root / "configs/external_versions.yaml"
    )

    with versions_path.open("r", encoding="utf-8") as handle:
        external_versions = yaml.safe_load(handle)

    for dependency_name, dependency in (
        external_versions["external_dependencies"].items()
    ):
        repository = project_root / dependency["local_path"]

        repository_exists = (
            repository.is_dir()
            and (repository / ".git").exists()
        )

        add_check(
            f"{dependency_name}_repository_exists",
            repository_exists,
            str(repository),
        )

        if repository_exists:
            actual_commit = git_commit(repository)
            expected_commit = dependency["commit"]

            add_check(
                f"{dependency_name}_commit",
                actual_commit == expected_commit,
                {
                    "expected": expected_commit,
                    "actual": actual_commit,
                },
            )

    disk_usage = shutil.disk_usage(project_root)
    free_gib = disk_usage.free / (1024**3)

    minimum_free_gib = 2.0

    add_check(
        "minimum_free_storage",
        free_gib >= minimum_free_gib,
        {
            "free_gib": round(free_gib, 3),
            "minimum_required_gib": minimum_free_gib,
        },
    )

    if free_gib < 8.0:
        warnings.append(
            "Available storage is below 8 GiB. "
            "Temporary frame extraction must remain limited."
        )

    return {
        "schema_version": 1,
        "stage": "input_validation",
        "dataset_id": configuration["dataset"]["id"],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration": str(configuration_path),
        "valid": len(errors) == 0,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "storage": {
            "free_bytes": disk_usage.free,
            "free_gib": round(free_gib, 3),
        },
    }
