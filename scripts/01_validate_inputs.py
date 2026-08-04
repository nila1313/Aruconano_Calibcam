#!/usr/bin/env python3
"""Run Stage 1 input validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SOURCE_ROOT))

from aruconano_calibcam.stages.validate_inputs import (  # noqa: E402
    validate_inputs,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate calibration pipeline inputs."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Dataset configuration YAML file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON validation report.",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    report = validate_inputs(
        project_root=PROJECT_ROOT,
        configuration_path=arguments.config,
    )

    output_path = arguments.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print("Input validation")
    print("================")
    print(f"Dataset: {report['dataset_id']}")
    print(f"Valid:   {report['valid']}")

    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}")

    for warning in report["warnings"]:
        print(f"[WARNING] {warning}")

    print()
    print(f"Report: {output_path}")

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
