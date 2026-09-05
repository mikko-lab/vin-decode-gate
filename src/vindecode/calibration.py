"""Calibration.

An uncalibrated confidence score makes the gate's threshold arbitrary. If
0.6 does not mean "right about 60% of the time", then choosing 0.6 rather
than 0.7 is superstition, and the business cannot reason about the trade
between a wrong answer and a missing one.

Temperature scaling is the method used here: one scalar, fitted on a held-out
calibration split by minimising negative log-likelihood. Two reasons it wins
over per-class Platt or isotonic on this corpus:

- With 32 model classes and roughly 1 350 usable VINs, several classes have
  two or three examples. Per-class calibration needs its own folds per class
  and simply refuses to fit -- scikit-learn's CalibratedClassifierCV raises
  on exactly that here. One shared parameter does not care.
- Temperature scaling is monotone in the logits, so it cannot change which
  label is predicted. Accuracy is untouched by construction; only the
  reported confidence moves. That keeps calibration and modelling separable.

Measured on this corpus, the classifier is *under*confident (mean confidence
0.76 against 0.86 accuracy), so the fitted temperature sits below 1 and
sharpens rather than softens.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def as_logits(scores: np.ndarray) -> np.ndarray:
    """Normalise decision_function output to a 2-D logit matrix.

    scikit-learn returns one score per sample for a two-class problem and one
    per class otherwise. Feeding the binary form straight to softmax would
    yield a 1x1 matrix -- probability 1.0 for the first class, every time,
    with no way to tell a confident decode from a coin flip. The two-class
    case is therefore expanded to [0, score], which is the equivalent
    logit pair.
    """
    array = np.asarray(scores, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[1] == 1:
        return np.hstack([np.zeros_like(array), array])
    return array


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, classes: np.ndarray) -> float:
    """Fit the scalar temperature that minimises NLL on a calibration split."""
    logits = as_logits(logits)
    indices = np.searchsorted(classes, y_true)

    def negative_log_likelihood(log_t: float) -> float:
        probabilities = softmax(logits / np.exp(log_t), axis=1)
        picked = probabilities[np.arange(len(indices)), indices]
        return float(-np.log(np.clip(picked, 1e-12, None)).mean())

    result = minimize_scalar(negative_log_likelihood, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Turn decision-function scores into calibrated probabilities."""
    return softmax(as_logits(logits) / temperature, axis=1)


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    """Mean gap between stated confidence and observed accuracy, bin-weighted.

    0 is perfect. This is the number that says whether a threshold means
    anything.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.sum():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def reliability_table(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 10
) -> list[dict]:
    """Per-bin breakdown behind the ECE, for reporting."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    table = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.sum():
            continue
        table.append(
            {
                "bin": f"{low:.1f}-{high:.1f}",
                "n": int(in_bin.sum()),
                "mean_confidence": round(float(confidence[in_bin].mean()), 3),
                "accuracy": round(float(correct[in_bin].mean()), 3),
            }
        )
    return table
