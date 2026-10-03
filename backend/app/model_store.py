from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from .config import (
    DETECT_MODEL,
    DETECT_MODEL_URL,
    DISEASE_MODEL,
    DISEASE_MODEL_DATA,
    DISEASE_MODEL_DATA_URL,
    DISEASE_MODEL_URL,
    GATE_MODEL,
    GATE_MODEL_DATA,
    GATE_MODEL_DATA_URL,
    GATE_MODEL_URL,
)


def normalize_download_url(url: str) -> str:
    if "huggingface.co" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    return url


def download_if_needed(url: str | None, destination: Path) -> None:
    if not url:
        return
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".download")
    urlretrieve(normalize_download_url(url), tmp_path)
    tmp_path.replace(destination)


def ensure_models() -> dict[str, bool]:
    download_if_needed(GATE_MODEL_URL, GATE_MODEL)
    download_if_needed(GATE_MODEL_DATA_URL, GATE_MODEL_DATA)
    download_if_needed(DETECT_MODEL_URL, DETECT_MODEL)
    download_if_needed(DISEASE_MODEL_URL, DISEASE_MODEL)
    download_if_needed(DISEASE_MODEL_DATA_URL, DISEASE_MODEL_DATA)

    return {
        "gate": GATE_MODEL.exists(),
        "gate_data": GATE_MODEL_DATA.exists(),
        "detector": DETECT_MODEL.exists(),
        "disease": DISEASE_MODEL.exists(),
        "disease_data": DISEASE_MODEL_DATA.exists(),
    }


if __name__ == "__main__":
    status = ensure_models()
    for name, exists in status.items():
        print(f"{name}: {'ok' if exists else 'missing'}")
    if not all(status.values()):
        raise SystemExit(1)
