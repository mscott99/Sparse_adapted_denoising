import pytest
import numpy as np
import scipy
from numpy.typing import NDArray
import pandas as pd
from scipy.fft import fft2

from sparse_recov.sampling_recov_functions import SF2_func, SF2_transpose_map, soft_threshold, Optimized_Bernoulli_prob_weights
from sparse_recov.uneven_sparse_experiment import threshold_vec, truncate_to_sparsity
from sparse_recov.subspace_prior import compute_local_coherences

from sparse_recov import experiment, make_df

def test_experiment():
    df = make_df()
    df.attrs['max_iter'] = 2
    random_rows = df.sample(n=30, random_state=42)  # random_state for reproducibility
    for _, row in random_rows.iterrows():
        experiment({**df.attrs, **row})  # test for runtime errors


def test_threshold_vec():
    # Test case 1: Basic functionality with k=2
    v = np.array([1, -5, 3, 2])
    k = 2
    result = threshold_vec(v, k)
    # Should keep -5 and 3 (the largest in absolute value)
    expected = np.array([0, -5, 3, 0])
    np.testing.assert_array_equal(result, expected)

    # Test case 2: 2D array
    v_2d = np.array([[1, -5], [3, 2]])
    k = 2
    result = threshold_vec(v_2d, k)
    expected = np.array([[0, -5], [3, 0]])
    np.testing.assert_array_equal(result, expected)

    # Test case 3: k equals array size (should return same array)
    v = np.array([1, 2, 3])
    k = 3
    result = threshold_vec(v, k)
    np.testing.assert_array_equal(result, v)

    # Test case 5: Check if it handles ties correctly
    v = np.array([1, 1, 1])
    k = 2
    result = threshold_vec(v, k)
    assert np.sum(result != 0) == k  # Should keep exactly k elements

    # Test case 6: Test with negative numbers
    v = np.array([-3, -2, -1])
    k = 1
    result = threshold_vec(v, k)
    expected = np.array([-3, 0, 0])
    np.testing.assert_array_equal(result, expected)


def test_truncate_to_sparsity():
    """Test the truncate_to_sparsity function"""
    # Create a simple test image
    image = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ])
    
    # Define simple transform and inverse transform functions
    # For testing, we'll use a simple "transform" that flattens the array
    def to_wavelet(img):
        return img.flatten()
    
    def from_wavelet(coeffs):
        return coeffs.reshape(image.shape)
    
    # Test with k=5 (keep 5 largest coefficients)
    k = 5
    result = truncate_to_sparsity(image, k, to_wavelet, from_wavelet)
    
    # The output should be a 4x4 array
    assert result.shape == (4, 4)
    
    # The output should have exactly k non-zero elements
    assert np.count_nonzero(result) == k
    
    # The positions of the non-zero elements should match the k largest values
    # in the original image
    largest_indices = np.argsort(np.abs(image.flatten()))[-k:]
    result_flat = result.flatten()
    
    # All the largest elements should be preserved
    for idx in largest_indices:
        assert result_flat[idx] != 0
    
    # Values that weren't in the top k should be zero
    for idx in range(16):
        if idx not in largest_indices:
            assert result_flat[idx] == 0


def test_compute_local_coherences():
    """Test the compute_local_coherences function"""
    # Use a small square matrix for testing
    sqt_n = 4
    n = sqt_n**2
    
    # Create a simple random matrix for Q
    np.random.seed(42)
    Q = np.random.randn(n//2, n)  # Using half the size for Q
    
    # Compute the local coherences
    local_coherences = compute_local_coherences(sqt_n, Q)
    
    # Check the shape of the output
    assert local_coherences.shape == (sqt_n, sqt_n)
    
    # Coherence values should be between 0 and 1
    assert np.all(local_coherences >= 0)
    assert np.all(local_coherences <= 1)
    
    # Manually verify a few coherence values
    # Create an orthonormal basis for the range of Q.T
    V = scipy.linalg.orth(Q.T)
    
    # Check a specific Fourier basis vector
    i, j = 0, 0  # First Fourier basis vector
    f = np.zeros((sqt_n, sqt_n), dtype=np.complex128)
    f[i, j] = 1
    f_vec = fft2(f, norm="ortho").ravel()
    
    # Compute projection norm onto range of Q.T
    proj_norm = np.linalg.norm(V @ (V.T.conj() @ f_vec))
    
    # Compare with the computed value
    assert np.isclose(local_coherences[i, j], proj_norm)

