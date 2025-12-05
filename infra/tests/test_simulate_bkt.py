import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os
import tempfile
from scripts.simulate_bkt import simulate_one_run, compute_metrics

def test_bkt_trend_correct_vs_incorrect():
    # small deterministic simulation to test trend: corrected behavior means mean delta after correct > mean delta after incorrect
    n_students = 20
    n_steps = 100
    n_concepts = 5
    seed = 12345
    params_per_concept = {f"c{i+1}": {"p_trans": 0.15, "p_guess": 0.2, "p_slip": 0.1} for i in range(n_concepts)}

    logs, final_mastery = simulate_one_run(n_students, n_steps, n_concepts, params_per_concept, seed=seed)
    metrics = compute_metrics(logs)

    # sanity: we expect positive average delta for correct answers
    assert metrics["n_correct"] > 0
    assert metrics["mean_delta_correct"] > metrics["mean_delta_incorrect"], (
        "Mean delta after correct answers should be larger than after incorrect answers"
    )
