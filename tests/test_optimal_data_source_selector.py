# tests/test_optimal_data_source_selector.py
import numpy as np
import pytest
from projektor.projector import OptimalDataSourceSelector, project_onto_simplex

def test_project_onto_simplex():
    # Test that projection maps any vector to a valid simplex.
    v = np.array([0.2, -0.1, 0.5, 0.4])
    w = project_onto_simplex(v)
    assert np.all(w >= 0)
    np.testing.assert_almost_equal(w.sum(), 1.0)

def test_optimal_data_source_selector():
    # Create dummy data for 4 sources.
    k = 4
    q_init = np.array([0.25, 0.25, 0.25, 0.25])
    # Create dummy dual solutions and lengths.
    # For simplicity, use random numbers.
    dual0 = np.random.rand(100)
    dual1 = np.random.rand(100)
    ds_lengths0 = [25, 25, 25, 25]
    ds_lengths1 = [25, 25, 25, 25]
    data_n0 = {'dual': dual0, 'ds_lengths': ds_lengths0, 'full_len': 100, 'ot_dist': 1.0}
    data_n1 = {'dual': dual1, 'ds_lengths': ds_lengths1, 'full_len': 100, 'ot_dist': 1.2}
    # Use default parameters (zeros)
    selector = OptimalDataSourceSelector(lr=1e-3, iterations=50, decay=0.95)
    optimal_q = selector.optimize(n=10000, n0=1000, n1=1500,
                                  data_n0=data_n0, data_n1=data_n1,
                                  q_init=q_init)
    # Check that the optimal mixing ratios sum to 1 and are nonnegative.
    np.testing.assert_almost_equal(optimal_q.sum(), 1.0)
    assert np.all(optimal_q >= 0)

