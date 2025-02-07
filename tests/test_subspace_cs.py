import pytest
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import datetime
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
from typing import TypedDict, Literal, Any, Tuple, List
from skimage.data import shepp_logan_phantom, camera
from skimage import transform
from pywt import wavedec2, waverec2, ravel_coeffs, unravel_coeffs, coeffs_to_array, array_to_coeffs
from utils import logspace, allcombinations, add_standard_attributes, run_local_df_experiment, repeat, soft_astype_df
from scipy.fft import fftshift, fft2

from sparse_recov.sampling_recov_functions import SF2_func, SF2_transpose_map, soft_threshold, Bernoulli_prob_weights
from sparse_recov.uneven_sparse_experiment import threshold_vec 

from sparse_recov import experiment,  make_df

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

