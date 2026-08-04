"""Stage 2A: calculate frame selections without extracting images."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aruconano_calibcam.config import load_configuration


class FramePlanningError(RuntimeError):
    """Raised when a valid stereo frame plan cannot be created."""


def build_frame_plan(
    project_root: Path,
    configuration_path: Path,
) -> dict[str, Any]:
    """Build a synchronized frame plan for all configured cameras."""

    project_root = project_root.resolve()
    configuration_path = configuration_path.resolve()

    configuration = load_configuration(configuration_path)

    cameras = configuration["cameras"]
    selection = configuration["frame_selection"]

    start = int(selection["start"])
    configured_end = selection["end"]
    step = int(selection["step"])
    offsets = [int(value) for value in selection["camera_offsets"]]

    if step <= 0:
        raise FramePlanningError(
            f"Frame step must be positive, received {step}."
        )

    if start < 0:
        raise FramePlanningError(
            f"Frame start must be non-negative, received {start}."
        )

    if len(offsets) != len(cameras):
        raise FramePlanningError(
            "The number of camera offsets must match "
            "the number of cameras."
        )

    # A base-frame index must remain valid after applying every
    # camera-specific offset.
    maximum_base_end = min(
        int(camera["frame_count"]) - offset
        for camera, offset in zip(cameras, offsets)
    )

    if configured_end is None:
        effective_end = maximum_base_end
    else:
        effective_end = min(
            int(configured_end),
            maximum_base_end,
        )

    if effective_end <= start:
        raise FramePlanningError(
            "The effective frame range is empty."
        )

    base_frame_indices = list(
        range(start, effective_end, step)
    )

    if not base_frame_indices:
        raise FramePlanningError(
            "Frame selection produced no indices."
        )

    camera_plans: list[dict[str, Any]] = []

    for camera, offset in zip(cameras, offsets):
        frame_count = int(camera["frame_count"])

        selected_indices = [
            base_index + offset
            for base_index in base_frame_indices
        ]

        invalid_indices = [
            index
            for index in selected_indices
            if index < 0 or index >= frame_count
        ]

        if invalid_indices:
            raise FramePlanningError(
                f"Camera {camera['index']} contains invalid "
                f"planned indices: {invalid_indices[:10]}"
            )

        camera_plans.append(
            {
                "camera_index": int(camera["index"]),
                "camera_name": str(camera["name"]),
                "video": str(camera["video"]),
                "frame_count": frame_count,
                "offset": offset,
                "selected_count": len(selected_indices),
                "first_selected_frame": selected_indices[0],
                "last_selected_frame": selected_indices[-1],
                "selected_frame_indices": selected_indices,
            }
        )

    return {
        "schema_version": 1,
        "stage": "frame_planning",
        "dataset_id": configuration["dataset"]["id"],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration": str(configuration_path),
        "selection": {
            "start": start,
            "configured_end": configured_end,
            "effective_end_exclusive": effective_end,
            "step": step,
            "base_selected_count": len(base_frame_indices),
            "base_first_frame": base_frame_indices[0],
            "base_last_frame": base_frame_indices[-1],
            "base_frame_indices": base_frame_indices,
        },
        "cameras": camera_plans,
        "total_images_if_both_cameras_extracted": sum(
            plan["selected_count"]
            for plan in camera_plans
        ),
        "storage_policy": {
            "extract_one_camera_at_a_time": True,
            "write_annotated_frames": False,
            "delete_temporary_frames_after_detection": True,
            "preserve_original_videos": True,
        },
        "note": (
            "Selected-frame count is not the same as valid-detection "
            "count. Frames without sufficient markers may be rejected "
            "during detection or ChArUco interpolation."
        ),
    }
