# tests/test_projector.py
import numpy as np
import pytest
from projektor.projector import Projector

@pytest.fixture
def sample_data():
    # Create sample experimental logs.
    # Each list element is a 1-D array of length < pad_length (we choose pad_length=11).
    reserrlog = [np.array([0.1, 0.2, 0.3]) for _ in range(5)]
    otlog = [np.array([1.0, 1.2, 1.1, 1.3]) for _ in range(5)]
    accs = [np.array([0.75, 0.78, 0.80, 0.82, 0.83]) for _ in range(5)]
    return reserrlog, otlog, accs

def test_align_data(sample_data):
    reserrlog, _, _ = sample_data
    proj = Projector(pad_length=11, threshold=5)
    aligned = proj.align_data(reserrlog)
    # Check that each row has length equal to pad_length.
    assert aligned.shape == (5, 11)
    # Check that the first three values match the original and the rest are zeros.
    np.testing.assert_array_almost_equal(aligned[0, :3], reserrlog[0])
    np.testing.assert_array_equal(aligned[0, 3:], np.zeros(8))

def test_compute_masks(sample_data):
    reserrlog, _, _ = sample_data
    proj = Projector(pad_length=11, threshold=2)
    aligned = proj.align_data(reserrlog)
    mask_low, mask_high = proj.compute_masks(aligned)
    # In our simple sample, since each row has 3 nonzero values,
    # and threshold=2, mask_low should be True for index 0,1 and mask_high True for index 2 if (i+j <=10).
    # Check shape and basic properties.
    assert mask_low.shape == (5, 11)
    assert mask_high.shape == (5, 11)
    # For the first row, first two indices should be True.
    assert mask_low[0, 0] == True
    assert mask_low[0, 1] == True
    # And index 2 should be in mask_high.
    assert mask_high[0, 2] == True

def test_compute_scaling_matrices():
    proj = Projector(pad_length=11, threshold=5)
    A1, A2 = proj.compute_scaling_matrices(11, 11)
    # Check that A1 and A2 have shape (11, 11)
    assert A1.shape == (11, 11)
    assert A2.shape == (11, 11)
    # Check boundary values.
    np.testing.assert_allclose(A1[0, 0], 0.0)
    np.testing.assert_allclose(A1[-1, 0], 1.0)
    np.testing.assert_allclose(A2[0, -1], 1.0)

def test_prepare_features(sample_data):
    reserrlog, otlog, accs = sample_data
    proj = Projector(pad_length=11, threshold=2)
    aligned = proj.align_data(reserrlog)
    _, mask_high = proj.compute_masks(aligned)
    X, y = proj.prepare_features(reserrlog, otlog, accs, mask_high)
    # We expect X to have shape (num_nonzero, 7) and y to be a 1-D array.
    assert X.ndim == 2
    assert X.shape[1] == 7
    assert y.ndim == 1

def test_fit_and_predict(sample_data):
    reserrlog, otlog, accs = sample_data
    proj = Projector(pad_length=11, threshold=2)
    aligned = proj.align_data(reserrlog)
    _, mask_high = proj.compute_masks(aligned)
    X, y = proj.prepare_features(reserrlog, otlog, accs, mask_high)
    result = proj.fit(X, y, verbose=False, plot=False)
    # Check that the fitted model is stored.
    assert proj.model is not None
    # Predict for one sample.
    pred = proj.predict(X[0])
    # The predicted value should be a finite float.
    assert np.isfinite(pred)

