#!/usr/bin/env python3
"""
models/bkt.py

Bayesian Knowledge Tracing (BKT) core functions.

Exposed functions:
- predict_correct(p, params) -> probability the student answers correctly
- bkt_update(p, correct, params) -> updated mastery probability after observing "correct" (bool)

Parameters (passed as dict `params`):
- p_trans : float  (learning/transition probability)  e.g. 0.15
- p_guess : float  (probability of correct if not mastered) e.g. 0.2
- p_slip  : float  (probability of incorrect despite mastery) e.g. 0.1

Notes:
- All probabilities are clamped to [0.0, 1.0].
- Implementation follows the classical BKT formulation:
    Pr(correct) = p*(1 - slip) + (1 - p)*guess
    Posterior given correct = p*(1 - slip) / Pr(correct)
    Posterior given incorrect = p*slip / Pr(incorrect)
    Then apply transition (learning) step:
    p_next = posterior + (1 - posterior) * p_trans
"""

from typing import Dict
import math


def _clamp(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _get_param(params: Dict, key: str, default: float) -> float:
    val = params.get(key, default) if isinstance(params, dict) else default
    try:
        return _clamp(float(val))
    except Exception:
        return default


def predict_correct(p: float, params: Dict) -> float:
    """
    Probability student answers correctly given mastery p and params.

    Args:
      p: current mastery probability (0..1)
      params: dict with keys 'p_guess', 'p_slip' (optional; defaults below)

    Returns:
      probability in [0,1]
    """
    p = _clamp(p)
    guess = _get_param(params, "p_guess", 0.2)
    slip = _get_param(params, "p_slip", 0.1)

    # Pr(correct) = p*(1 - slip) + (1 - p)*guess
    pr = p * (1.0 - slip) + (1.0 - p) * guess
    return _clamp(pr)


def bkt_update(p: float, correct: bool, params: Dict) -> float:
    """
    Perform one BKT update: posterior given observation, then apply transition.

    Args:
      p: prior mastery probability (0..1)
      correct: whether the student's response was correct (True/False)
      params: dict with keys:
          - p_trans (learning probability after the attempt)
          - p_guess
          - p_slip

    Returns:
      updated mastery probability (0..1)
    """
    p = _clamp(p)
    guess = _get_param(params, "p_guess", 0.2)
    slip = _get_param(params, "p_slip", 0.1)
    p_trans = _get_param(params, "p_trans", 0.15)

    # Likelihoods
    pr_correct = p * (1.0 - slip) + (1.0 - p) * guess
    pr_incorrect = 1.0 - pr_correct

    # Avoid division by zero — handle degenerate cases
    if correct:
        if pr_correct <= 0.0:
            posterior = 0.0
        else:
            posterior = (p * (1.0 - slip)) / pr_correct
    else:
        if pr_incorrect <= 0.0:
            posterior = 0.0
        else:
            posterior = (p * slip) / pr_incorrect

    posterior = _clamp(posterior)

    # Transition (learning between opportunities)
    updated = posterior + (1.0 - posterior) * p_trans
    updated = _clamp(updated)
    return updated
