from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from .config import (
    DETECT_MODEL,
    DISEASE_CLASS_NAMES,
    DISEASE_MODEL,
    GATE_CLASS_NAMES,
    GATE_MODEL,
    IMAGE_SIZE,
    MEAN,
    STD,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits -= np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def center_crop_resize(image: Image.Image, img_size: int = IMAGE_SIZE) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    scale = 256 / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)
    left = (resized.width - img_size) // 2
    top = (resized.height - img_size) // 2
    cropped = resized.crop((left, top, left + img_size, top + img_size))

    arr = np.asarray(cropped).astype(np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    height, width = arr.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(arr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_width) / 2
    pad_y = (size - new_height) / 2
    left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    canvas[top : top + new_height, left : left + new_width] = resized

    tensor = canvas.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor.astype(np.float32), scale, (left, top)


@dataclass
class OnnxClassifier:
    model_path: Path
    class_names: list[str]

    def __post_init__(self) -> None:
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image: Image.Image) -> dict[str, Any]:
        logits = self.session.run([self.output_name], {self.input_name: center_crop_resize(image)})[0]
        probs = softmax(np.squeeze(logits))
        best_idx = int(np.argmax(probs))
        probabilities = {name: float(probs[idx]) for idx, name in enumerate(self.class_names)}
        return {
            "prediction": self.class_names[best_idx],
            "confidence": float(probs[best_idx]),
            "probabilities": probabilities,
        }


@dataclass
class YoloOnnxDetector:
    model_path: Path
    input_size: int | None = None

    def __post_init__(self) -> None:
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_name = self.session.get_outputs()[0].name
        if self.input_size is None:
            height, width = model_input.shape[2], model_input.shape[3]
            self.input_size = int(height if isinstance(height, int) else width)

    def detect(
        self,
        image: Image.Image,
        conf: float = 0.25,
        iou: float = 0.45,
        max_detections: int = 3,
    ) -> list[dict[str, Any]]:
        tensor, scale, (pad_x, pad_y) = letterbox(image, int(self.input_size))
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        preds = np.squeeze(raw)
        if preds.ndim != 2:
            return []
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes: list[list[int]] = []
        boxes_xyxy: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        width, height = image.size

        for pred in preds:
            if pred.shape[0] < 5:
                continue
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores)) if class_scores.size else 0
            score = float(class_scores[class_id]) if class_scores.size else float(pred[4])
            if score < conf:
                continue

            cx, cy, w, h = [float(v) for v in pred[:4]]
            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale
            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_xyxy.append([x1, y1, x2, y2])
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(score)
            class_ids.append(class_id)

        keep = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
        if len(keep) == 0:
            return []

        kept_indices = np.array(keep).reshape(-1).tolist()
        kept_indices = sorted(kept_indices, key=lambda idx: scores[idx], reverse=True)[:max_detections]
        return [
            {
                "rank": rank,
                "class_id": class_ids[idx],
                "class_name": "tomato_leaf",
                "detect_confidence": scores[idx],
                "box_xyxy": boxes_xyxy[idx],
            }
            for rank, idx in enumerate(kept_indices, start=1)
        ]


def crop_with_padding(image: Image.Image, box: list[float], padding: int = 8) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box
    return image.crop(
        (
            max(0, int(x1) - padding),
            max(0, int(y1) - padding),
            min(width, int(x2) + padding),
            min(height, int(y2) + padding),
        )
    )


class TomatoPipeline:
    def __init__(self) -> None:
        self.gate = OnnxClassifier(GATE_MODEL, GATE_CLASS_NAMES)
        self.detector = YoloOnnxDetector(DETECT_MODEL)
        self.disease = OnnxClassifier(DISEASE_MODEL, DISEASE_CLASS_NAMES)

    def predict(
        self,
        image: Image.Image,
        gate_threshold: float = 0.60,
        detect_conf: float = 0.25,
        detect_iou: float = 0.45,
        max_detections: int = 3,
        include_probabilities: bool = False,
    ) -> dict[str, Any]:
        gate_result = self.gate.predict(image)
        gate_result["threshold"] = gate_threshold
        result: dict[str, Any] = {
            "gate": gate_result,
            "detections": [],
            "crop_predictions": [],
            "final_status": "not_tomato_leaf",
            "message": "Input image did not pass the tomato leaf gate.",
        }

        if gate_result["probabilities"].get("tomato_leaf", 0.0) < gate_threshold:
            return result

        detections = self.detector.detect(image, detect_conf, detect_iou, max_detections)
        result["detections"] = detections
        if not detections:
            result["final_status"] = "no_leaf_detected"
            result["message"] = "Tomato leaf gate passed, but YOLO found no leaf boxes."
            return result

        crop_predictions: list[dict[str, Any]] = []
        for detection in detections:
            crop = crop_with_padding(image, detection["box_xyxy"])
            prediction = self.disease.predict(crop)
            entry = {
                "rank": detection["rank"],
                "box_xyxy": detection["box_xyxy"],
                "detect_confidence": detection["detect_confidence"],
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
            }
            if include_probabilities:
                entry["probabilities"] = prediction["probabilities"]
            crop_predictions.append(entry)

        best = max(crop_predictions, key=lambda item: item["confidence"])
        result.update(
            {
                "crop_predictions": crop_predictions,
                "best_disease_prediction": best["prediction"],
                "best_disease_confidence": best["confidence"],
                "final_status": "tomato_leaf_classified",
                "message": "Tomato leaf disease classified from detected crop.",
            }
        )
        return result
