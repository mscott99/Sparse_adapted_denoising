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
from typing import TypedDict, Literal, Any, Tuple, List
from skimage.data import shepp_logan_phantom, camera
from skimage import transform
from pywt import wavedec2, waverec2, ravel_coeffs, unravel_coeffs, coeffs_to_array, array_to_coeffs
from utils import logspace, allcombinations, add_standard_attributes, run_local_df_experiment, repeat, soft_astype_df
from scipy.fft import fftshift, fft2

from src.sampling_recov_functions import SF2_func, SF2_transpose_map, soft_threshold, Bernoulli_prob_weights

from src.uneven_sparse_experiment import experiment, lasso_Fourier_haar_basis, truncate_to_sparsity, make_df

# def test_uneven_recov():
#     ExperimentTypes = {"sqt_n": int,'max_iter': int, 'k': int, "m": int, "m_tld": int, "sigma": float, "lambda_": float,
#                    "tol": float, "img": pd.CategoricalDtype(["phantom", "camera"]), 'experiment_fn_label': str, "sampling_type": pd.CategoricalDtype(["wr", "Ber"]), "emp_snr": float}
#     def make_local_df():
#         sqt_n = 8
#         np.random.seed(0)
#         df = allcombinations({'sqt_n': sqt_n, 'm': int(sqt_n**2/2),'max_iter':5, 'k':3, 'lambda_': 0.001, 'sigma': 0.00, 'tol': 1e-5,
#                              "img": "camera", "sampling_type": "Ber", "adapted_sampling": True, 'save_img': True})
#         # df = repeat(df, 2)
#         df['seed'] = [np.random.randint(1, 2**32 - 1) for _ in range(len(df))]
#         add_standard_attributes(df)
#         df.attrs.update({"global_seed": 0, "level":4})
#         return soft_astype_df(df, ExperimentTypes)
#
#     local_df = make_local_df()
#     local_df['seed']
#     result = run_local_df_experiment(local_df, experiment, dtype=ExperimentTypes)


def test_experiment():
    df = make_df()
    experiment({**df.attrs, **df.loc[0]})  # test for runtime errors


# def truncation():
#     # It seems like truncated coefficients are out of the range of the transformation.
#     # Upon inverting, it meddles with things, but then stabilizes afterwards.
#
#     sqt_n = 16
#     true_signal = transform.resize(
#         camera().astype(np.float32), (sqt_n, sqt_n))
#
#     k = 10
#     level = 1
#     sample_wavelet, slices = coeffs_to_array(
#         wavedec2(true_signal, "haar", level=level))
#     theshape = sample_wavelet.shape
#
#     sample_wavelet, slices = coeffs_to_array(
#         pywt.wavedec2(true_signal, "haar", level=level))
#
#     theshape = sample_wavelet.shape
#
#     def wavelet_transform(y) -> np.ndarray:  # 2d to 1d
#         return coeffs_to_array(wavedec2(y, "haar", level=level))[0].ravel()
#
#     def inverse_wavelet(w: np.ndarray) -> np.ndarray:  # 1d to 2d
#         return waverec2(array_to_coeffs(
#             w.reshape(theshape), slices, output_format="wavedec2"), "haar")
#
#     truncated = truncate_to_sparsity(
#         true_signal, k, wavelet_transform, inverse_wavelet)
#
#     show_vec_mag(wavelet_transform(truncated))
#     # It works.
#
#     plt.imshow(true_signal)
#     plt.imshow(truncated)
