import numpy as np
import cv2


def get_mouth_mask_without_poisson(frame_shape, landmarks):
    h, w = frame_shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mouth_points = landmarks[52:72].astype(np.int32)
    x, y, mw, mh = cv2.boundingRect(mouth_points)

    padding_h = int(mh * 0.4)
    padding_w = int(mw * 0.3)

    x = max(0, x - padding_w)
    y = max(0, y - padding_h)
    mw = min(w - x, mw + 2 * padding_w)
    mh = min(h - y, mh + 2 * padding_h)

    mouth_box = np.array(
        [[x, y], [x + mw, y], [x + mw, y + mh], [x, y + mh]], dtype=np.int32
    )

    cv2.fillConvexPoly(mask, mouth_box, 0)

    feather = max(5, min(mw, mh) // 10)
    if feather % 2 == 0:
        feather += 1

    for _ in range(2):
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)

    return mask


def get_mouth_mask_with_poisson(frame_shape, landmarks):
    if landmarks is None or len(landmarks) < 106:
        return None

    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    lower_lip_order = [
        65,
        66,
        62,
        70,
        69,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        0,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        65,
    ]
    mouth_points = landmarks[52:72].astype(np.int32)
    x, y, mw, mh = cv2.boundingRect(mouth_points)

    padding_h = int(mh * 0.2)
    padding_w = int(mw * 0.1)

    x = max(0, x - padding_w)
    y = max(0, y - padding_h)
    mw = min(w - x, mw + 2 * padding_w)
    mh = min(h - y, mh + 2 * padding_h)

    mouth_box = np.array(
        [[x, y], [x + mw, y], [x + mw, y + mh], [x, y + mh]],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(mask, mouth_box, 255)

    feather = max(5, min(mw, mh) // 10)
    if feather % 2 == 0:
        feather += 1

    for _ in range(2):
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)

    return mask


def get_mask_center(mask, frame_shape):
    """Get center"""
    if mask is None:
        return None

    h, w = frame_shape[:2]
    M = cv2.moments(mask)

    if M["m00"] == 0:
        h, w = mask.shape[:2]
        return (w // 2, h // 2)

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    cx = max(5, min(w - 5, cx))
    cy = max(5, min(h - 5, cy))
    return (cx, cy)
