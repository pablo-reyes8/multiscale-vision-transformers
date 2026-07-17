"""Unified inference and analysis utilities."""

from .explain import find_last_conv2d, grad_cam, occlusion_sensitivity
from .metrics import (
    calibration_stats,
    classification_report,
    collect_predictions,
    confusion_matrix,
    evaluate_loader,
)

__all__ = [
    "calibration_stats",
    "classification_report",
    "collect_predictions",
    "confusion_matrix",
    "evaluate_loader",
    "find_last_conv2d",
    "grad_cam",
    "occlusion_sensitivity",
]
