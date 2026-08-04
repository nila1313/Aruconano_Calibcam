"""Stage 2B: extract planned frames from one camera recording."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from aruconano_calibcam.config import load_configuration


class FrameExtractionError(RuntimeError):
    """Raised when selected frames cannot be extracted safely."""


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


def load_frame_plan(path: Path) -> dict[str, Any]:
    """Load a generated frame-plan JSON file."""

    path = path.resolve()

    if not path.is_file():
        raise FrameExtractionError(
            f"Frame plan does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)

    if plan.get("stage") != "frame_planning":
        raise FrameExtractionError(
            f"Unexpected frame-plan stage: {plan.get('stage')}"
        )

    return plan


def resolve_camera(
    configuration: dict[str, Any],
    plan: dict[str, Any],
    camera_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve matching camera entries from configuration and plan."""

    configured = [
        camera
        for camera in configuration["cameras"]
        if camera["name"] == camera_name
    ]

    planned = [
        camera
        for camera in plan["cameras"]
        if camera["camera_name"] == camera_name
    ]

    if len(configured) != 1 or len(planned) != 1:
        raise FrameExtractionError(
            f"Could not uniquely resolve camera: {camera_name}"
        )

    return configured[0], planned[0]


def extract_camera_frames(
    project_root: Path,
    configuration_path: Path,
    frame_plan_path: Path,
    camera_name: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Extract all planned frames for one configured camera."""

    project_root = project_root.resolve()
    output_directory = output_directory.resolve()

    configuration = load_configuration(configuration_path)
    plan = load_frame_plan(frame_plan_path)

    if plan["dataset_id"] != configuration["dataset"]["id"]:
        raise FrameExtractionError(
            "Dataset configuration and frame plan do not match."
        )

    camera, camera_plan = resolve_camera(
        configuration,
        plan,
        camera_name,
    )

    video_path = project_root / camera["video"]

    if not video_path.is_file():
        raise FrameExtractionError(
            f"Video does not exist: {video_path}"
        )

    if output_directory.exists():
        raise FrameExtractionError(
            f"Output directory already exists: {output_directory}"
        )

    selected_indices = [
        int(index)
        for index in camera_plan["selected_frame_indices"]
    ]

    if not selected_indices:
        raise FrameExtractionError(
            "The frame plan contains no selected indices."
        )

    if selected_indices != sorted(set(selected_indices)):
        raise FrameExtractionError(
            "Selected frame indices must be sorted and unique."
        )

    disk_before = shutil.disk_usage(project_root)

    minimum_free_bytes = 2 * 1024**3

    if disk_before.free < minimum_free_bytes:
        raise FrameExtractionError(
            "Less than 2 GiB of free storage is available."
        )

    output_directory.mkdir(parents=True, exist_ok=False)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        output_directory.rmdir()
        raise FrameExtractionError(
            f"OpenCV could not open video: {video_path}"
        )

    target_set = set(selected_indices)
    last_target = selected_indices[-1]

    extracted_files: list[dict[str, Any]] = []
    frame_index = 0

    try:
        while frame_index <= last_target:
            success, frame = capture.read()

            if not success or frame is None:
                raise FrameExtractionError(
                    f"Video decoding failed at frame {frame_index}."
                )

            if frame_index in target_set:
                filename = f"frame_{frame_index:06d}.png"
                output_path = output_directory / filename
                temporary_path = (
                    output_directory
                    / f".{filename}.temporary.png"
                )

                write_success = cv2.imwrite(
                    str(temporary_path),
                    frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )

                if not write_success:
                    raise FrameExtractionError(
                        f"Could not write frame: {temporary_path}"
                    )

                temporary_path.replace(output_path)

                extracted_files.append(
                    {
                        "frame_index": frame_index,
                        "filename": filename,
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                        "channels": (
                            int(frame.shape[2])
                            if frame.ndim == 3
                            else 1
                        ),
                        "size_bytes": output_path.stat().st_size,
                        "sha256": sha256(output_path),
                    }
                )

                if len(extracted_files) % 20 == 0:
                    print(
                        f"Extracted {len(extracted_files)} / "
                        f"{len(selected_indices)} frames"
                    )

            frame_index += 1

    except Exception:
        capture.release()
        raise

    capture.release()

    extracted_indices = [
        item["frame_index"]
        for item in extracted_files
    ]

    if extracted_indices != selected_indices:
        missing = sorted(
            set(selected_indices) - set(extracted_indices)
        )

        raise FrameExtractionError(
            "Extracted indices do not match the frame plan. "
            f"Missing: {missing[:20]}"
        )

    total_size_bytes = sum(
        item["size_bytes"]
        for item in extracted_files
    )

    disk_after = shutil.disk_usage(project_root)

    return {
        "schema_version": 1,
        "stage": "frame_extraction",
        "dataset_id": configuration["dataset"]["id"],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration": str(
            Path(configuration_path).resolve()
        ),
        "frame_plan": str(
            Path(frame_plan_path).resolve()
        ),
        "camera": {
            "index": int(camera["index"]),
            "name": str(camera["name"]),
            "video": str(video_path),
            "video_sha256": str(camera["sha256"]),
        },
        "output_directory": str(output_directory),
        "planned_count": len(selected_indices),
        "extracted_count": len(extracted_files),
        "first_frame": extracted_indices[0],
        "last_frame": extracted_indices[-1],
        "total_size_bytes": total_size_bytes,
        "files": extracted_files,
        "storage": {
            "free_before_bytes": disk_before.free,
            "free_after_bytes": disk_after.free,
            "consumed_bytes": (
                disk_before.free - disk_after.free
            ),
        },
        "valid": extracted_indices == selected_indices,
    }
