from pathlib import Path
from typing import Literal

# ---------------- Paths ----------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


ANALYSIS_MODEL = "buffalo_l"

SWAPPER_MODEL = (
    MODELS_DIR / "inswapper_128.onnx"
)  # higher model inswapper_128_fp16.onnx

ENHANCER_MODEL = MODELS_DIR / "GFPGANv1.4.onnx"
ENHANCE_WEIGHT = 0.6  # blend strength: 0 = full GFPGAN, 1 = original face

INSIGHTFACE_DIR = Path.home() / ".insightface" / "models"
BUFFALO_L_DIR = INSIGHTFACE_DIR / "buffalo_l"

# ---------------- Application Settings ----------------

# Camera settings
DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Face detection settings
FACE_DETECTION_SIZE = (640, 640)
FACE_CONFIDENCE_THRESHOLD = 0.5
