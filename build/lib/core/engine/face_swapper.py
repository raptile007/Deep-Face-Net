import os
import cv2
import insightface
from core.face_analyser import get_face_analyser
from core.config import SWAPPER_MODEL

swapper_ = None


def get_face_swapper():
    global swapper_
    if swapper_ is None:
        swapper_ = insightface.model_zoo.get_model(str(SWAPPER_MODEL))
    return swapper_


def swap_face(source_face, target_face, frame):
    face_swapper = get_face_swapper()

    return face_swapper.get(frame, target_face, source_face, paste_back=True)
