from pathlib import Path
from typing import Literal

# ---------------- Paths ----------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


ANALYSIS_MODEL = "buffalo_l"

SWAPPER_MODEL = (
    MODELS_DIR / "inswapper_128.onnx"
)  # higher model inswapper_128_fp16.onnx

# ---------------- Application Settings ----------------

# Camera settings
DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Face detection settings
FACE_DETECTION_SIZE = (640, 640)
FACE_CONFIDENCE_THRESHOLD = 0.5
