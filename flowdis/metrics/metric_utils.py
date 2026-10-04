"""Dichotomous-segmentation scores reported by FlowDIS.

Predictions are floats in ``[0, 1]``. Targets are binarized at 0.5. ``beta2`` is
the β² used by the F-measure papers: 0.3 for max F-measure, 1 for weighted
F-measure. Dataset scores for max F-measure and mean E-measure come from the
threshold curve averaged over images, which is what ``SegmentationMetrics`` returns.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import convolve, distance_transform_edt
from torch import Tensor

_EPS = np.spacing(1)


def _pair(prediction: Tensor, target: Tensor) -> tuple[np.ndarray, np.ndarray]:
    pred = prediction.detach().float().cpu().numpy()
    gt = target.detach().float().cpu().numpy()
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction shape {pred.shape} does not match target shape {gt.shape}")
    if pred.ndim != 2:
        raise ValueError(f"expected one mask, got shape {pred.shape}")
    return np.clip(pred, 0.0, 1.0).astype(np.float64), gt >= 0.5


def _f_curve(prediction: np.ndarray, target: np.ndarray, beta2: float) -> np.ndarray:
    levels = (prediction * 255).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    foreground, _ = np.histogram(levels[target], bins=bins)
    background, _ = np.histogram(levels[~target], bins=bins)
    true_positive = np.cumsum(np.flip(foreground))
    predicted_positive = true_positive + np.cumsum(np.flip(background))
    predicted_positive[predicted_positive == 0] = 1
    positive = max(int(np.count_nonzero(target)), 1)
    precision = true_positive / predicted_positive
    recall = true_positive / positive
    numerator = (1 + beta2) * precision * recall
    denominator = np.where(numerator == 0, 1, beta2 * precision + recall)
    return numerator / denominator


def max_f_measure(prediction: Tensor, target: Tensor, beta2: float = 0.3) -> float:
    """Maximum F-measure over thresholds 0…255."""
    pred, gt = _pair(prediction, target)
    return float(_f_curve(pred, gt, beta2).max())


def mean_absolute_error(prediction: Tensor, target: Tensor) -> float:
    pred, gt = _pair(prediction, target)
    return float(np.mean(np.abs(pred - gt)))


def _gaussian_kernel(size: int = 7, sigma: float = 5.0) -> np.ndarray:
    radius = (size - 1) / 2
    coords = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(coords[0] ** 2 + coords[1] ** 2) / (2 * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    total = kernel.sum()
    if total != 0:
        kernel /= total
    return kernel


def weighted_f_measure(prediction: Tensor, target: Tensor, beta2: float = 1.0) -> float:
    """Weighted F-measure. An empty foreground scores 0."""
    pred, gt = _pair(prediction, target)
    if not np.any(gt):
        return 0.0

    distance, indices = distance_transform_edt(gt == 0, return_indices=True)
    error = np.abs(pred - gt.astype(np.float64))
    nearest_foreground_error = error.copy()
    background = gt == 0
    nearest_foreground_error[background] = error[indices[0][background], indices[1][background]]
    smoothed = convolve(nearest_foreground_error, weights=_gaussian_kernel(), mode="constant", cval=0.0)
    dependent = np.where(gt & (smoothed < error), smoothed, error)
    importance = np.where(background, 2 - np.exp(np.log(0.5) / 5 * distance), 1.0)
    weighted_error = dependent * importance

    true_positive = np.sum(gt) - np.sum(weighted_error[gt])
    false_positive = np.sum(weighted_error[background])
    recall = 1 - np.mean(weighted_error[gt])
    precision = true_positive / (true_positive + false_positive + _EPS)
    return float((1 + beta2) * recall * precision / (recall + beta2 * precision + _EPS))


def _object_score(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return 2 * mean / (mean**2 + 1 + std + _EPS)


def _region_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size == 0:
        return 0.0
    count = prediction.size
    mean_prediction = float(np.mean(prediction))
    mean_target = float(np.mean(target))
    sigma_prediction = np.sum((prediction - mean_prediction) ** 2) / (count - 1 + _EPS)
    sigma_target = np.sum((target - mean_target) ** 2) / (count - 1 + _EPS)
    sigma_cross = np.sum((prediction - mean_prediction) * (target - mean_target)) / (count - 1 + _EPS)
    numerator = 4 * mean_prediction * mean_target * sigma_cross
    denominator = (mean_prediction**2 + mean_target**2) * (sigma_prediction + sigma_target)
    if numerator != 0:
        return float(numerator / (denominator + _EPS))
    if denominator == 0:
        return 1.0
    return 0.0


def structure_measure(prediction: Tensor, target: Tensor, alpha: float = 0.5) -> float:
    """Structure-measure, Sα. ``alpha`` weights the object term against the region term."""
    pred, gt = _pair(prediction, target)
    foreground_ratio = float(np.mean(gt))
    if foreground_ratio == 0:
        return float(1 - np.mean(pred))
    if foreground_ratio == 1:
        return float(np.mean(pred))

    object_score = _object_score(pred[gt]) * foreground_ratio
    object_score += _object_score((1 - pred)[~gt]) * (1 - foreground_ratio)

    height, width = gt.shape
    if np.count_nonzero(gt) == 0:
        center_y, center_x = round(height / 2), round(width / 2)
    else:
        center_y, center_x = np.argwhere(gt).mean(axis=0).round()
    center_y, center_x = int(center_y) + 1, int(center_x) + 1
    area = height * width
    quadrants = (
        (pred[:center_y, :center_x], gt[:center_y, :center_x], center_x * center_y / area),
        (pred[:center_y, center_x:], gt[:center_y, center_x:], center_y * (width - center_x) / area),
        (pred[center_y:, :center_x], gt[center_y:, :center_x], (height - center_y) * center_x / area),
        (pred[center_y:, center_x:], gt[center_y:, center_x:], None),
    )
    weights = [item[2] for item in quadrants[:3]]
    weights.append(1 - sum(weights))
    region_score = sum(_region_ssim(part, mask) * weight for (part, mask, _), weight in zip(quadrants, weights))
    return float(max(0.0, alpha * object_score + (1 - alpha) * region_score))


def _alignment_parts(foreground_hit, foreground_miss, predicted_foreground, predicted_background, target_foreground, size):
    background_miss = target_foreground - foreground_hit
    background_hit = predicted_background - background_miss
    counts = [foreground_hit, foreground_miss, background_miss, background_hit]
    mean_prediction = predicted_foreground / size
    mean_target = target_foreground / size
    pairs = (
        (1 - mean_prediction, 1 - mean_target),
        (1 - mean_prediction, 0 - mean_target),
        (0 - mean_prediction, 1 - mean_target),
        (0 - mean_prediction, 0 - mean_target),
    )
    return counts, pairs


def _e_curve(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    size = target.size
    target_foreground = int(np.count_nonzero(target))
    levels = (prediction * 255).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    hit_hist, _ = np.histogram(levels[target], bins=bins)
    miss_hist, _ = np.histogram(levels[~target], bins=bins)
    foreground_hit = np.cumsum(np.flip(hit_hist))
    foreground_miss = np.cumsum(np.flip(miss_hist))
    predicted_foreground = foreground_hit + foreground_miss
    predicted_background = size - predicted_foreground

    if target_foreground == 0:
        enhanced = predicted_background
    elif target_foreground == size:
        enhanced = predicted_foreground
    else:
        counts, pairs = _alignment_parts(
            foreground_hit,
            foreground_miss,
            predicted_foreground,
            predicted_background,
            target_foreground,
            size,
        )
        enhanced = np.zeros(256, dtype=np.float64)
        for count, (prediction_value, target_value) in zip(counts, pairs):
            alignment = 2 * prediction_value * target_value / (prediction_value**2 + target_value**2 + _EPS)
            enhanced += ((alignment + 1) ** 2) / 4 * count
    return enhanced / (size - 1 + _EPS)


def e_measure(prediction: Tensor, target: Tensor) -> float:
    """Mean E-measure over thresholds 0…255."""
    pred, gt = _pair(prediction, target)
    return float(_e_curve(pred, gt).mean())


class SegmentationMetrics:
    """Running dataset scores. Call ``update`` once per image, then ``compute``."""

    def __init__(self, beta2: float = 0.3, alpha: float = 0.5):
        self.beta2 = beta2
        self.alpha = alpha
        self._mae: list[float] = []
        self._weighted_f: list[float] = []
        self._structure: list[float] = []
        self._f_curves: list[np.ndarray] = []
        self._e_curves: list[np.ndarray] = []

    def update(self, prediction: Tensor, target: Tensor) -> None:
        prediction = prediction.detach().float().cpu()
        target = target.detach().float().cpu()
        if prediction.ndim == 4:
            prediction = prediction[:, 0]
        if target.ndim == 4:
            target = target[:, 0]
        if prediction.ndim == 2:
            prediction = prediction.unsqueeze(0)
            target = target.unsqueeze(0)
        for index in range(prediction.shape[0]):
            pred, gt = _pair(prediction[index], target[index])
            self._mae.append(float(np.mean(np.abs(pred - gt))))
            self._weighted_f.append(weighted_f_measure(prediction[index], target[index]))
            self._structure.append(structure_measure(prediction[index], target[index], alpha=self.alpha))
            self._f_curves.append(_f_curve(pred, gt, self.beta2))
            self._e_curves.append(_e_curve(pred, gt))

    def compute(self) -> dict[str, float]:
        if not self._mae:
            raise RuntimeError("SegmentationMetrics.update was not called")
        f_curve = np.mean(self._f_curves, axis=0)
        e_curve = np.mean(self._e_curves, axis=0)
        return {
            "weighted_f_measure": float(np.mean(self._weighted_f)),
            "max_f_measure": float(f_curve.max()),
            "mae": float(np.mean(self._mae)),
            "structure_measure": float(np.mean(self._structure)),
            "e_measure": float(e_curve.mean()),
        }
