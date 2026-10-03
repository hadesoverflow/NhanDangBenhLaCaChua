from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/tomato-onnx-models"))

LOCAL_GATE_MODEL = REPO_ROOT / "model_classify_leaf_domain" / "resnet50_tomato_leaf_gate.onnx"
LOCAL_GATE_MODEL_DATA = REPO_ROOT / "model_classify_leaf_domain" / "resnet50_tomato_leaf_gate.onnx.data"
LOCAL_DETECT_MODEL = REPO_ROOT / "model_detect_leaf_tomato" / "model_detect_plant_leaf.onnx"
LOCAL_DISEASE_MODEL = REPO_ROOT / "model_classify_disease_tomato" / "resnet50_tomato.onnx"
LOCAL_DISEASE_MODEL_DATA = REPO_ROOT / "model_classify_disease_tomato" / "resnet50_tomato.onnx.data"

GATE_MODEL_URL = os.environ.get("GATE_MODEL_URL")
GATE_MODEL_DATA_URL = os.environ.get("GATE_MODEL_DATA_URL")
DETECT_MODEL_URL = os.environ.get("DETECT_MODEL_URL")
DISEASE_MODEL_URL = os.environ.get("DISEASE_MODEL_URL")
DISEASE_MODEL_DATA_URL = os.environ.get("DISEASE_MODEL_DATA_URL")

GATE_MODEL = MODEL_CACHE_DIR / "resnet50_tomato_leaf_gate.onnx" if GATE_MODEL_URL else LOCAL_GATE_MODEL
GATE_MODEL_DATA = MODEL_CACHE_DIR / "resnet50_tomato_leaf_gate.onnx.data" if GATE_MODEL_DATA_URL else LOCAL_GATE_MODEL_DATA
DETECT_MODEL = MODEL_CACHE_DIR / "model_detect_plant_leaf.onnx" if DETECT_MODEL_URL else LOCAL_DETECT_MODEL
DISEASE_MODEL = MODEL_CACHE_DIR / "resnet50_tomato.onnx" if DISEASE_MODEL_URL else LOCAL_DISEASE_MODEL
DISEASE_MODEL_DATA = MODEL_CACHE_DIR / "resnet50_tomato.onnx.data" if DISEASE_MODEL_DATA_URL else LOCAL_DISEASE_MODEL_DATA

GATE_CLASS_NAMES = ["tomato_leaf", "out_of_domain"]
DISEASE_CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
