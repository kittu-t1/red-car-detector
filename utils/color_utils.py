# utils/color_utils.py

import cv2
import numpy as np

# Supported color names
COLOR_NAMES = ["red", "blue", "black", "white", "green", "yellow", "silver"]

# TIGHTENED HSV ranges for each color
COLOR_HSV_RANGES = {
    "red": [
        # Only very saturated, bright reds ⇒ H around 0–10 or 170–180, S≥150, V≥100
        ([0, 150, 100], [10, 255, 255]),
        ([170, 150, 100], [180, 255, 255])
    ],
    "blue": [
        ([90, 120, 70], [130, 255, 255])    # unchanged “standard blue” 
    ],
    "black": [
        # Narrow “true black” ⇒ V ≤ 80 AND S ≤ 50 (rejects dark gray/road/shadows whose V often >80 or S >50)
        ([0, 0, 0], [180, 255, 50])
    ],
    "white": [
        ([0, 0, 220], [180, 30, 255])       # unchanged “bright white (low saturation)”
    ],
    "green": [
        ([35, 80, 50], [90, 255, 255])      # unchanged “broad green”
    ],
    "yellow": [
        ([20, 100, 100], [40, 255, 255])    # unchanged “bright yellow”
    ],
    "silver": [
        # Silver/gray = low saturation (≤30) and mid‐to‐high V (100–200)
        ([0, 0, 100], [180, 30, 200])
    ]
}


def get_color_ratio(image: np.ndarray, color: str) -> float:
    
    """
    Returns the fraction of pixels in `image` (BGR) that lie within the HSV ranges for `color`.
    If the crop is empty or the color name is invalid, returns 0.0.
    """
    
    if image is None or image.size == 0:
        return 0.0

    color = color.lower()
    if color not in COLOR_HSV_RANGES:
        return 0.0

    # 1) Smooth the image to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2) Build a combined mask for all sub‐ranges of this color
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in COLOR_HSV_RANGES[color]:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        submask = cv2.inRange(hsv, lower_np, upper_np)
        mask = cv2.bitwise_or(mask, submask)

    # 3) Morphological opening/closing to remove small noise islands
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4) Compute ratio
    total_pixels = mask.shape[0] * mask.shape[1]
    if total_pixels == 0:
        return 0.0

    pixel_count = cv2.countNonZero(mask)
    return pixel_count / float(total_pixels)
