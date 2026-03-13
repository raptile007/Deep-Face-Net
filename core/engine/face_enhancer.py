"""
Face enhancement engine using GFPGAN.
Enhances only the face crop (not the full frame) for speed and quality.
"""

import numpy as np
import cv2
import threading
from pathlib import Path

from core.config import ENHANCER_MODEL, ENHANCE_WEIGHT

_ENHANCER = None
_LOCK = threading.Lock()


def get_face_enhancer():
    """Lazy singleton loader for the ONNX GFPGAN model."""
    global _ENHANCER
    if _ENHANCER is None:
        with _LOCK:
            if _ENHANCER is None:
                if not Path(ENHANCER_MODEL).exists():
                    raise FileNotFoundError(
                        f"GFPGAN model not found at: {ENHANCER_MODEL}\n"
                        "Run: python download_models.py --model GFPGANv1.4.onnx"
                    )
                try:
                    import onnxruntime
                except ImportError:
                    raise ImportError("onnxruntime is not installed.")

                providers = onnxruntime.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

                # Simple ONNX wrapper mimicking GFPGANer.enhance
                class ONNXEnhancer:
                    def __init__(self, model_path, providers):
                        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
                        self.input_name = self.session.get_inputs()[0].name
                        self.output_name = self.session.get_outputs()[0].name

                    def enhance(self, img, has_aligned=True, only_center_face=True, paste_back=True, weight=0.5):
                        # img is a 512x512 BGR image
                        # BGR to RGB, normalize to [-1, 1], CHW
                        img_rgb = img[:, :, ::-1]
                        img_norm = img_rgb.astype(np.float32) / 255.0
                        img_norm = (img_norm - 0.5) / 0.5
                        img_norm = np.transpose(img_norm, (2, 0, 1))
                        img_norm = np.expand_dims(img_norm, axis=0)

                        # Inference
                        out = self.session.run([self.output_name], {self.input_name: img_norm})[0][0]

                        # CHW to HWC, denormalize to [0, 255], RGB to BGR
                        out = np.transpose(out, (1, 2, 0))
                        out = np.clip((out + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
                        out_bgr = out[:, :, ::-1]

                        # Blend with original
                        if weight < 1.0:
                            out_bgr = cv2.addWeighted(img, 1.0 - weight, out_bgr, weight, 0)

                        return None, None, out_bgr

                _ENHANCER = ONNXEnhancer(str(ENHANCER_MODEL), providers)
    return _ENHANCER


def enhance_faces(frame: np.ndarray, faces: list, frame_size: tuple) -> np.ndarray:
    """
    Enhance each detected face in the frame using GFPGAN.

    For each face:
      1. Crop a 512×512 aligned face region (what GFPGAN expects natively).
      2. Run GFPGAN on just the crop — fast and high quality.
      3. Warp the enhanced crop back into the original frame space.
      4. Blend using a soft Gaussian-feathered elliptical mask.

    Args:
        frame:      BGR frame (H×W×3).
        faces:      List of InsightFace face objects (must have .kps attribute).
        frame_size: (width, height) of the frame — needed for warpAffine.

    Returns:
        Frame with enhanced faces blended back in (same shape as input).
    """
    from insightface.utils import face_align

    enhancer = get_face_enhancer()
    frame_w, frame_h = frame_size
    result = frame.copy()

    for face in faces:
        kps = face.kps  # 5-point landmarks

        # ── 1. Align + crop to 512×512 ──────────────────────────────────────
        aligned = face_align.norm_crop(frame, landmark=kps, image_size=512)

        # ── 2. Enhance with GFPGAN ──────────────────────────────────────────
        _, _, enhanced = enhancer.enhance(
            aligned,
            has_aligned=True,       # Skip internal detection; we pre-aligned
            only_center_face=True,
            paste_back=True,
            weight=ENHANCE_WEIGHT,
        )

        if enhanced is None:
            continue

        # ── 3. Warp enhanced crop back to original frame space ───────────────
        M = face_align.estimate_norm(kps, image_size=512)
        IM = cv2.invertAffineTransform(M)
        enhanced_back = cv2.warpAffine(enhanced, IM, (frame_w, frame_h))

        # ── 4. Soft elliptical mask for smooth blending ──────────────────────
        mask = np.zeros((512, 512), dtype=np.float32)
        cv2.ellipse(
            mask,
            center=(256, 256),
            axes=(220, 270),
            angle=0, startAngle=0, endAngle=360,
            color=1.0, thickness=-1,
        )
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        mask_back = cv2.warpAffine(mask, IM, (frame_w, frame_h))
        mask_back = mask_back[:, :, np.newaxis]  # H×W → H×W×1 for broadcasting

        result = (
            enhanced_back * mask_back + result * (1.0 - mask_back)
        ).astype(np.uint8)

    return result
