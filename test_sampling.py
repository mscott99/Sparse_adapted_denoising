# This file is for uneven signal recovery.
import pytest
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import datetime
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
from typing import TypedDict, NotRequired, Literal, Any, Tuple, List
from skimage.data import shepp_logan_phantom, camera
from skimage import transform
from pywt import wavedec2, waverec2, ravel_coeffs, unravel_coeffs, coeffs_to_array, array_to_coeffs
from utils import logspace, allcombinations, add_standard_attributes, run_local_df_experiment, repeat, soft_astype_df
from scipy.fft import fftshift, fft2

from src.sampling_recov_functions import SF2_func, SF2_transpose_map, soft_threshold, Bernoulli_prob_weights

from src.uneven_sparse_experiment import experiment, lasso_Fourier_haar_basis, truncate_to_sparsity

def test_uneven_recov():
    ExperimentTypes = {"sqt_n": int, "m": int, "m_tld": int, "sigma": float, "lambda_": float,
                   "tol": float, "img": pd.CategoricalDtype(["phantom", "camera"]), 'experiment_fn_label': str, "sampling_type": pd.CategoricalDtype(["wr", "Ber"]), "emp_snr": float}
    def make_local_df():
        sqt_n = 20
        np.random.seed(0)
        df = allcombinations({'sqt_n': sqt_n, 'm': int(sqt_n**2/2), 'lambda_': 0.001, 'sigma': 0.00, 'tol': 1e-5,
                             "img": "camera", "sampling_type": "Ber", "adapted_sampling": True, 'save_img': True})
        # df = repeat(df, 2)
        df['seed'] = [np.random.randint(1, 2**32 - 1) for _ in range(len(df))]
        add_standard_attributes(df)
        df.attrs.update({"global_seed": 0, "level":4})
        return soft_astype_df(df, ExperimentTypes)

    local_df = make_local_df()
    local_df['seed']
    result = run_local_df_experiment(local_df, experiment, dtype=ExperimentTypes)

def test_truncate_to_sparsity():
    # Test case 1: Simple test with known sparsity
    test_image = np.zeros((8, 8))
    test_image[0:4, 0:4] = 1  # Create a simple pattern
    
    # Test with different sparsity levels
    k_values = [1, 5, 10]
    
    for k in k_values:
        truncated = truncate_to_sparsity(test_image, k, level=2)
        
        # Check output dimensions
        assert truncated.shape == test_image.shape
        
        # Transform truncated image to check sparsity
        coeffs, _ = coeffs_to_array(wavedec2(truncated, 'haar', level=2))
        
        # Count non-zero coefficients
        non_zero_count = np.count_nonzero(coeffs)
        
        # Check if number of non-zero coefficients is less than or equal to k
        assert non_zero_count <= k, f"Expected {k} non-zero coefficients, got {non_zero_count}"
        
        # Check if output is real-valued
        assert np.isreal(truncated).all(), "Output contains complex values"

    # Test case 2: Test with real image
    test_image = camera()
    k = 1000
    truncated = truncate_to_sparsity(test_image, k, level=5)
    
    # Check if dimensions are preserved
    assert truncated.shape == test_image.shape
    
    # Check sparsity
    coeffs, _ = coeffs_to_array(wavedec2(truncated, 'haar', level=5))
    assert np.count_nonzero(coeffs) <= k

def truncation():
    # It seems like truncated coefficients are out of the range of the transformation.
    # Upon inverting, it meddles with things, but then stabilizes afterwards.
    sqt_n=32
    true_signal = transform.resize(
        shepp_logan_phantom().astype(np.float32), (sqt_n, sqt_n))

    sample_wavelet, slices = coeffs_to_array(wavedec2(true_signal, "haar", level=3))
    theshape = sample_wavelet.shape

    def wavelet_transform(y) -> np.ndarray: #2d to 1d
        return coeffs_to_array(wavedec2(y, "haar", level=3))[0].ravel()

    def inverse_wavelet(w: np.ndarray) -> np.ndarray: # 1d to 2d
        return waverec2(array_to_coeffs(
            w.reshape(theshape), slices, output_format="wavedec2"), "haar")

    truncated = truncate_to_sparsity(true_signal, sqt_n**2, wavelet_transform, inverse_wavelet)

    plt.imshow(true_signal)
    plt.imshow(truncated)

    insides = wavelet_transform(truncated)
    (np.sum(np.abs(insides))/np.linalg.norm(insides))**2
    (truncated - inverse_wavelet(insides)).max()
    plt.imshow(truncated - inverse_wavelet(insides))

    np.sum(insides > 1e-5)

    inverse_wavelet(insides)


    len(insides) 

    np.allclose(wavelet_transform(inverse_wavelet(insides)), insides)

    tester = np.random.randn(32, 32)

    test_had=np.random.randn(*wavelet_transform(tester).shape)

    np.allclose(wavelet_transform(inverse_wavelet(test_had)), test_had)


    np.allclose(inverse_wavelet(wavelet_transform(tester)), tester)

    np.sum(insides > 1e-4)
    # It does not seem to be exact

