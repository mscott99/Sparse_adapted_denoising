import pytest
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.fft import fft2

from sparse_recov.sampling_recov_functions import SF2_func, SF2_transpose_map, soft_threshold, Bernoulli_prob_weights, threshold_prob_weights
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
    assert np.all(np.diff(probs) >= 0)  # Probabilities should be non-decreasing
    
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

def test_threshold_prob_weights():
    """Test threshold_prob_weights function with various test cases"""
    
    # Test case 1: Basic functionality
    loc_co = np.array([0.1, 0.4, 0.2, 0.5, 0.3])
    num_robusts = 2
    
    result = threshold_prob_weights(num_robusts, loc_co)
    
    # Check return type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == loc_co.shape
    
    # Check that result is normalized (sum of squares should be 1)
    assert np.isclose(np.sum(result), 1)
    # Test case 2: Edge case with num_robusts = 0
    loc_co_2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    num_robusts_2 = 0
    
    result_2 = threshold_prob_weights(num_robusts_2, loc_co_2)
    
    # All values should be squared since num_robusts = 0
    assert np.allclose(result_2, loc_co_2**2 / np.sum(loc_co_2**2))
    
    # Test case 3: Edge case with num_robusts = len(loc_co)
    num_robusts_3 = len(loc_co_2)
    
    result_3 = threshold_prob_weights(num_robusts_3, loc_co_2)
    
    # No values should be squared, just normalized
    assert np.allclose(result_3, loc_co_2 / np.sum(loc_co_2))
    

