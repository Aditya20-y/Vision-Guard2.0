"""Scientific evaluation metrics (Image AUROC, Pixel AUROC, AUPRO, F1, Precision, Recall)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def compute_evaluation_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    masks: Optional[Sequence[np.ndarray]] = None,
    anomaly_maps: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, Optional[float]]:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)
    y_pred = (y_score > threshold).astype(np.int32)

    image_auroc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None
    pixel_auroc: Optional[float] = None
    if masks and anomaly_maps:
        m_flat = np.concatenate([m.ravel() for m in masks])
        s_flat = np.concatenate([a.ravel() for a in anomaly_maps])
        if len(np.unique(m_flat)) == 2:
            pixel_auroc = float(roc_auc_score(m_flat, s_flat))

    return {
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "image_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def apply_corruption(image: np.ndarray, corruption: str, severity: int, seed: int = 42) -> np.ndarray:
    """Deterministic industrial image degradations. Never modifies ground-truth masks."""
    import cv2

    sev = max(1, min(5, int(severity)))
    rng = np.random.default_rng(seed + sev * 997)
    img = image.astype(np.float32).copy()

    if corruption == "gaussian_noise":
        std = [0.03, 0.06, 0.10, 0.15, 0.22][sev - 1]
        return np.clip(img + rng.normal(0.0, std, size=img.shape), 0.0, 1.0).astype(np.float32)
    if corruption == "gaussian_blur":
        sigma = [0.6, 1.1, 1.7, 2.4, 3.2][sev - 1]
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    if corruption == "brightness_shift":
        delta = [0.08, 0.15, -0.20, 0.26, -0.34][sev - 1]
        return np.clip(img + delta, 0.0, 1.0).astype(np.float32)
    if corruption == "contrast_shift":
        factor = [0.82, 0.68, 0.54, 0.42, 0.30][sev - 1]
        return np.clip(0.5 + (img - 0.5) * factor, 0.0, 1.0).astype(np.float32)
    if corruption == "jpeg_compression":
        quality = [65, 42, 25, 14, 7][sev - 1]
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return img
        dec = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return (dec.astype(np.float32) / 255.0)
    if corruption == "resize_resample":
        scale = [0.75, 0.60, 0.48, 0.35, 0.25][sev - 1]
        h, w = img.shape[:2]
        sh, sw = max(8, int(h * scale)), max(8, int(w * scale))
        down = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"Unknown corruption type: {corruption}")


def assess_image_quality(image: np.ndarray) -> Dict[str, object]:
    """Independent Image Quality Assessment decoupled from defect classification."""
    import cv2

    u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_lum = float(np.mean(gray))
    rms_contrast = float(np.std(gray))
    h, w = gray.shape

    warnings: List[str] = []
    if blur_var < 35.0:
        warnings.append(f"Excessive blur detected (Laplacian variance {blur_var:.1f}).")
    if mean_lum < 35.0 or mean_lum > 225.0:
        warnings.append(f"Extreme illumination detected (mean luminance {mean_lum:.1f}/255).")
    if rms_contrast < 14.0:
        warnings.append(f"Very low image contrast (RMS contrast {rms_contrast:.1f}).")
    if min(h, w) < 128:
        warnings.append(f"Low input resolution ({w}x{h}px).")

    return {
        "overall_status": "acceptable" if not warnings else ("caution" if len(warnings) == 1 else "poor"),
        "warnings": warnings,
        "blur_variance": blur_var,
        "mean_luminance": mean_lum,
        "rms_contrast": rms_contrast,
    }
