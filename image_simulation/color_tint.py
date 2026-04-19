"""
Global color-tint transforms for simulating non-standard sensor color schemes.
Applied post-rendering to already-simulated images.
"""
import numpy as np

# Per-channel scale factors for each tint.
# Values chosen to produce a strong but non-destructive tint while preserving contrast.
_TINT_MATRICES = {
    "none":   np.array([1.00, 1.00, 1.00], dtype=np.float32),
    "orange": np.array([1.30, 1.05, 0.50], dtype=np.float32),
    "purple": np.array([1.20, 0.60, 1.30], dtype=np.float32),
}

TINT_NAMES = list(_TINT_MATRICES.keys())


def apply_tint(image: np.ndarray, tint: str) -> np.ndarray:
    """
    Apply a global color tint to an RGB image.

    :param image: uint8 numpy array of shape (H, W, 3) — RGB.
    :param tint: one of TINT_NAMES ("none", "orange", "purple").
    :return: uint8 tinted image, same shape.
    """
    if tint not in _TINT_MATRICES:
        raise ValueError(f"Unknown tint '{tint}'. Valid: {TINT_NAMES}")
    if tint == "none":
        return image
    scales = _TINT_MATRICES[tint]
    tinted = image.astype(np.float32) * scales
    return np.clip(tinted, 0, 255).astype(np.uint8)
