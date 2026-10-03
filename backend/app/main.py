from __future__ import annotations

from functools import lru_cache
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .inference import TomatoPipeline
from .model_store import ensure_models


app = FastAPI(title="Tomato Leaf Disease ONNX API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_pipeline() -> TomatoPipeline:
    model_status = ensure_models()
    if not all(model_status.values()):
        missing = [name for name, exists in model_status.items() if not exists]
        raise RuntimeError(f"Missing model files: {', '.join(missing)}")
    return TomatoPipeline()


@app.get("/health")
def health() -> dict[str, object]:
    models = ensure_models()
    return {"status": "ok" if all(models.values()) else "missing_model", "models": models}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    gate_threshold: float = Form(0.60),
    detect_conf: float = Form(0.25),
    detect_iou: float = Form(0.45),
    max_detections: int = Form(3),
    include_probabilities: bool = Form(False),
) -> dict[str, object]:
    try:
        image = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    try:
        return get_pipeline().predict(
            image=image,
            gate_threshold=gate_threshold,
            detect_conf=detect_conf,
            detect_iou=detect_iou,
            max_detections=max_detections,
            include_probabilities=include_probabilities,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
