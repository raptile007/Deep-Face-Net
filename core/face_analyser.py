import cv2
import insightface
import threading

from core.config import ANALYSIS_MODEL

# Global singleton instance
FACE_ANALYSER_ = None
LOCK_ = threading.Lock()


def get_face_analyser():
    global FACE_ANALYSER_

    if FACE_ANALYSER_ is None:
        with LOCK_:
            # Double-check locking pattern
            if FACE_ANALYSER_ is None:
                analyser = insightface.app.FaceAnalysis(name=ANALYSIS_MODEL)
                analyser.prepare(ctx_id=0, det_size=(640, 640))
                FACE_ANALYSER_ = analyser

    return FACE_ANALYSER_
