from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the ONNX tomato backend API.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--backend-url", default=os.environ.get("BACKEND_URL", "http://localhost:8010"))
    parser.add_argument("--gate-threshold", type=float, default=0.60)
    parser.add_argument("--detect-conf", type=float, default=0.25)
    parser.add_argument("--detect-iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=3)
    parser.add_argument("--include-probabilities", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.image.open("rb") as image_file:
        response = requests.post(
            f"{args.backend_url.rstrip('/')}/predict",
            files={"file": (args.image.name, image_file, "image/jpeg")},
            data={
                "gate_threshold": args.gate_threshold,
                "detect_conf": args.detect_conf,
                "detect_iou": args.detect_iou,
                "max_detections": args.max_detections,
                "include_probabilities": str(args.include_probabilities).lower(),
            },
            timeout=120,
        )
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
