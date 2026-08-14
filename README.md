# ArUco Nano + CalibCam Calibration Pipeline

This repository contains a reproducible stereo camera calibration workflow for
ChArUco-board recordings. It validates dataset metadata, plans synchronized
frames, extracts selected images, detects ArUco markers with ArUco Nano,
converts detections into CalibCam-compatible ChArUco YAML, and preserves the
selected calibration outputs.

The current repository also includes a final result package:

- `Aruconano_Calibcam_Final_Results.zip`
- `upload_package/Aruconano_Calibcam_Final_Results/`

The source videos and generated working directories are intentionally ignored by
Git because they are large.

## Repository Layout

- `configs/datasets/` - dataset definitions for each camera pair and tested
  frame/offset variants.
- `configs/experiments/` - small experiment configs used for offset tests.
- `configs/external_versions.yaml` - pinned external dependency versions.
- `resources/boards/` - canonical calibration board arrays.
- `scripts/` - numbered pipeline entry points.
- `src/aruconano_calibcam/` - Python implementation of the pipeline stages.
- `src/native/aruco_nano_detector.cpp` - native OpenCV/ArUco Nano detector.
- `upload_package/Aruconano_Calibcam_Final_Results/` - selected final
  calibration inputs, logs, summaries, and outputs.

## Requirements

The pipeline expects:

- Python 3
- OpenCV Python bindings
- NumPy
- PyYAML
- CalibCam / `calibcamlib`
- `clang++`
- `pkg-config`
- OpenCV 4 discoverable through `pkg-config`
- the pinned `external/aruco_nano/aruco_nano.h` header

Source calibration videos should be placed under the paths declared in each
dataset YAML, for example:

```text
calibration_videos/20230613/4pi/030_checkerboard_1/
```

## Pipeline

Use a dataset config from `configs/datasets/`. The examples below use
`configs/datasets/pair_01.yaml`.

1. Validate inputs:

```bash
python3 scripts/01_validate_inputs.py \
  --config configs/datasets/pair_01.yaml \
  --output runs/pair_01/01_validation.json
```

2. Plan synchronized frames:

```bash
python3 scripts/02_plan_frames.py \
  --config configs/datasets/pair_01.yaml \
  --output runs/pair_01/02_frame_plan.json
```

3. Extract selected frames for each camera:

```bash
python3 scripts/03_extract_frames.py \
  --config configs/datasets/pair_01.yaml \
  --plan runs/pair_01/02_frame_plan.json \
  --camera left \
  --output-dir runs/pair_01/frames_left \
  --manifest runs/pair_01/03_extract_left.json

python3 scripts/03_extract_frames.py \
  --config configs/datasets/pair_01.yaml \
  --plan runs/pair_01/02_frame_plan.json \
  --camera right \
  --output-dir runs/pair_01/frames_right \
  --manifest runs/pair_01/03_extract_right.json
```

4. Build the native detector:

```bash
scripts/04_build_aruco_nano_detector.sh
```

5. Detect ArUco markers:

```bash
python3 scripts/05_detect_markers.py \
  --config configs/datasets/pair_01.yaml \
  --extraction-manifest runs/pair_01/03_extract_left.json \
  --camera left \
  --detector build/native/aruco_nano_detector \
  --output-dir runs/pair_01/detections_left \
  --manifest runs/pair_01/05_detect_left.json

python3 scripts/05_detect_markers.py \
  --config configs/datasets/pair_01.yaml \
  --extraction-manifest runs/pair_01/03_extract_right.json \
  --camera right \
  --detector build/native/aruco_nano_detector \
  --output-dir runs/pair_01/detections_right \
  --manifest runs/pair_01/05_detect_right.json
```

6. Convert detections to CalibCam ChArUco YAML:

```bash
python3 scripts/06_convert_charuco.py \
  --config configs/datasets/pair_01.yaml \
  --extraction-manifest runs/pair_01/03_extract_left.json \
  --marker-manifest runs/pair_01/05_detect_left.json \
  --camera left \
  --output runs/pair_01/detection_000_left.yml \
  --manifest runs/pair_01/06_charuco_left.json

python3 scripts/06_convert_charuco.py \
  --config configs/datasets/pair_01.yaml \
  --extraction-manifest runs/pair_01/03_extract_right.json \
  --marker-manifest runs/pair_01/05_detect_right.json \
  --camera right \
  --output runs/pair_01/detection_001_right.yml \
  --manifest runs/pair_01/06_charuco_right.json
```

7. Optionally generate annotated ArUco and ChArUco visualizations:

```bash
python3 scripts/07_generate_visualizations.py \
  --config configs/datasets/pair_01.yaml \
  --extraction-manifest runs/pair_01/03_extract_left.json \
  --marker-manifest runs/pair_01/05_detect_left.json \
  --charuco-detection runs/pair_01/detection_000_left.yml \
  --camera left \
  --aruco-output-dir runs/pair_01/annotated_aruco_left \
  --charuco-output-dir runs/pair_01/annotated_charuco_left \
  --manifest runs/pair_01/07_visualizations_left.json
```

CalibCam can then be run with the generated `detection_000_left.yml`,
`detection_001_right.yml`, board file, and options file for the selected pair.

## Final Results

The final package contains selected calibration outputs for pairs 01, 02, 03,
04, 05, and 07. Pair 06 is included as a detection-only result and was rejected
for calibration because no markers were detected.

For each calibrated pair, the package includes:

- `inputs/` - board, options, and detection YAML files used by CalibCam.
- `results/` - single-camera and multi-camera calibration outputs.
- `calibcam.log` - CalibCam run log.
- selection summaries or metadata.

`SHA256SUMS.txt` and `FILE_INVENTORY.txt` are included in the result package to
verify file integrity and inspect the delivered contents.

## Git Notes

The `.gitignore` excludes source recordings, temporary frames, generated run
directories, external dependency checkouts, build products, and common Python or
macOS cache files. Keep source videos in `calibration_videos/` locally rather
than committing them.
