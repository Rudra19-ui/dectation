from typing import Tuple

import cv2
import numpy as np


def resize_image(image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def normalize_image(image: np.ndarray, method: str = "minmax") -> np.ndarray:
    if method == "minmax":
        return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    elif method == "meanstd":
        return (image - np.mean(image)) / (np.std(image) + 1e-8)
    else:
        raise ValueError("Unknown normalization method")


def segment_breast_region(image: np.ndarray) -> np.ndarray:
    # Placeholder: simple thresholding; replace with U-Net for advanced segmentation
    _, mask = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(image, image, mask=mask)


def preprocess_pipeline(
    image: np.ndarray, size: Tuple[int, int] = (224, 224), norm_method: str = "minmax"
) -> np.ndarray:
    image = resize_image(image, size)
    image = apply_clahe(image)
    image = segment_breast_region(image)
    image = normalize_image(image, method=norm_method)
    return image
