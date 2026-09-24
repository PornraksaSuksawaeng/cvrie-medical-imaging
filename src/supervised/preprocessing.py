import numpy as np
from pathlib import Path
import cv2


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a 3D RGB image to a 2D grayscale image using the luminance formula."""
    if len(image.shape) == 3:
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        image = 0.299 * r + 0.587 * g + 0.114 * b  # Gray = 0.299×R + 0.587×G + 0.114×B
    return image


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a single image to the target (width, height)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def center_crop(image: np.ndarray, ratio: float = 0.7) -> np.ndarray:
    """Keep only the central part of the image, based on a ratio of its size."""
    h, w = image.shape
    new_h, new_w = int(h * ratio), int(w * ratio)

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    return image[top : top + new_h, left : left + new_w]


def normalize(image: np.ndarray) -> np.ndarray:
    """Scale pixel values to the 0-1 range."""
    return image / 255.0


def save_processed(X, y, path: str | Path) -> None:
    """Save the processed images."""
    pass
