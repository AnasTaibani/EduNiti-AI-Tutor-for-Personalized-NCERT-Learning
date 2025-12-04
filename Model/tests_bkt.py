#!/usr/bin/env python3
# tests/test_bkt.py

import pytest
from bkt import predict_correct, bkt_update

DEFAULT_PARAMS = {"p_trans": 0.15, "p_guess": 0.2, "p_slip": 0.1}


def test_predict_correct_range_and_formula():
    p = 0.3
    pr = predict_correct(p, DEFAULT_PARAMS)
    # Manual calc
    expected = p * (1 - DEFAULT_PARAMS["p_slip"]) + (1 - p) * DEFAULT_PARAMS["p_guess"]
    assert pytest.approx(expected, rel=1e-6) == pr
    assert 0.0 <= pr <= 1.0


def test_bkt_increases_after_correct_sequence():
    p = 0.2
    params = {"p_trans": 0.2, "p_guess": 0.2, "p_slip": 0.1}
    seq = [True] * 5  # five consecutive correct responses
    prev = p
    for obs in seq:
        new_p = bkt_update(prev, obs, params)
        # Should not decrease after a correct response
        assert new_p >= prev - 1e-9
        prev = new_p
    # After repeated corrects, mastery should approach 1 (not necessarily reach)
    assert prev > p


def test_bkt_decreases_after_incorrect_sequence():
    p = 0.8
    params = {"p_trans": 0.05, "p_guess": 0.2, "p_slip": 0.1}
    seq = [False] * 3
    prev = p
    for obs in seq:
        new_p = bkt_update(prev, obs, params)
        # Posterior before transition often decreases; after low p_trans it should still be <= prev
        assert new_p <= prev + 1e-9
        prev = new_p
    assert prev < p


def test_edge_cases_and_clamping():
    # p out of range -> clamped
    params = {"p_trans": 0.1, "p_guess": 0.0, "p_slip": 0.0}
    p = -0.5
    new_p = bkt_update(p, True, params)
    assert 0.0 <= new_p <= 1.0

    p = 1.5
    new_p = bkt_update(p, False, params)
    assert 0.0 <= new_p <= 1.0
