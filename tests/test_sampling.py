import pytest
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.fft import fft2

from sparse_recov.sampling_recov_functions import SF2_func, SF2_transpose_map, soft_threshold, Bernoulli_prob_weights, sample_wor
from sparse_recov import experiment, make_df


def test_experiment():
    df = make_df()
    experiment({**df.attrs, **df.loc[0]})  # test for runtime errors


def test_bernoulli_prob_weights_basic():
    """Test basic functionality of Bernoulli_prob_weights"""
    # Create a simple coherence vector with all positive values
    loc_co = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    m = 3  # We want to sample 3 measurements

    probs, J, Lsqr = Bernoulli_prob_weights(loc_co, m)

    # Check return types
    assert isinstance(probs, np.ndarray)
    assert isinstance(J, int)
    assert isinstance(Lsqr, float)

    # Check that probabilities are between 0 and 1
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)

    # Check that higher coherence values have higher sampling probabilities
    # Probabilities should be non-decreasing
    assert np.all(np.diff(probs) >= 0)

    # Check that the expected number of measurements is close to m
    assert np.isclose(np.sum(probs), m, rtol=0.1)


def test_bernoulli_prob_weights_with_zeros():
    """Test Bernoulli_prob_weights with some zero coherence values"""
    # Create a coherence vector with some zeros
    loc_co = np.array([0.0, 0.0, 0.3, 0.4, 0.5])
    m = 2  # We want to sample 2 measurements

    probs, J, Lsqr = Bernoulli_prob_weights(loc_co, m)

    # Check that zero coherence values have zero sampling probabilities
    assert probs[0] == 0
    assert probs[1] == 0

    # Check that the expected number of measurements is close to m
    # Only counting the non-zero positions
    assert np.isclose(np.sum(probs), m, rtol=0.1)

    probs2, J, Lsqr = Bernoulli_prob_weights(loc_co, 3)
    print(probs2)
    assert np.isclose(probs2, np.array([0, 0, 1, 1, 1])).all()


def test_wor():
    """Test that all indices are sampled exactly once (no duplicates, no missing)."""
    p = [0.4, 0.3, 0.2, 0.1]
    indices, counts = sample_wor(p, 4)
    expected_indices = np.arange(len(p))
    assert np.array_equal(np.sort(
        indices), expected_indices), f"Expected all indices {expected_indices}, got {indices}"
    assert len(indices) == len(
        p), f"Expected {len(p)} indices, got {len(indices)}"

    """Test that all counts are at least 1 (each index was sampled at least once)."""
    p = [0.4, 0.3, 0.2, 0.05, 0.05]
    indices, counts = sample_wor(p, 4)
    assert len(indices) == 4
    assert np.sum(counts) >= 4

    """Test with highly skewed probabilities to ensure progress and correctness."""
    p = [0.99, 0.005, 0.003, 0.002]
    indices, counts = sample_wor(p, 4)
    assert len(indices) == 4
    assert np.array_equal(np.sort(
        indices), expected_indices), f"Expected all indices {expected_indices}, got {indices}"
    assert np.all(counts >= 1), f"Expected all counts >= 1, got {counts}"
    assert len(indices) == len(
        p), f"Expected {len(p)} indices, got {len(indices)}"

    p= [0.999, 0.000999, 0.000001]
    assert np.all(sample_wor(p, 2)[0] == [0,1])

    p= np.repeat(1e-3, 1000)
    indices, counts = sample_wor(p, int(9e2))
    assert len(indices) == int(9e2)

    # Create a skewed probability distribution
    n = 1000
    p = np.array([0.9999] + [0.0001 / (n - 1)] * (n - 1))  # One high probability, rest very low
    p /= p.sum()  # Normalize to sum to 1
    m = 50  # Requesting many distinct indices, likely to take too long

    # Check that ValueError is raised
    with pytest.raises(ValueError):
        sample_wor(p, m)

def test_bernoulli_prob_weights_edge_cases():
    """Test Bernoulli_prob_weights with edge cases"""
    # Case 1: All coherence values are equal
    loc_co = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    m = 3

    probs, J, Lsqr = Bernoulli_prob_weights(loc_co, m)

    # All probabilities should be equal in this case
    assert np.allclose(probs, probs[0])

    # Check that the expected number of measurements is close to m
    assert np.isclose(np.sum(probs), m, rtol=0.1)

    # Case 2: One very large coherence value, others small
    loc_co = np.array([0.1, 0.1, 0.1, 0.1, 10.0])
    m = 2

    probs, J, Lsqr = Bernoulli_prob_weights(loc_co, m)

    # The last position should have probability close to 1
    assert probs[-1] > 0.9

    # Check that the expected number of measurements is close to m
    assert np.isclose(np.sum(probs), m, rtol=0.1)


def test_bernoulli_prob_weights_assertion():
    """Test that Bernoulli_prob_weights raises an assertion error when m is too large"""
    # Create a coherence vector with many zeros
    loc_co = np.array([0.0, 0.0, 0.0, 0.4, 0.5])
    m = 3  # More than the number of non-zero coherence values

    # Should raise an assertion error
    with pytest.raises(AssertionError):
        Bernoulli_prob_weights(loc_co, m)
