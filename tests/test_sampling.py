import pytest
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.fft import fft2

from sparse_recov.sampling_recov_functions import SF2_func, SF2_transpose_map, soft_threshold, Bernoulli_prob_weights
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

    probs2, J, Lsqr = Bernoulli_prob_weights(loc_co, 3)
    print(probs2)
    assert np.isclose(probs2, np.array([0,0,1,1,1])).all()


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
