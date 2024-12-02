# This file is for uneven signal recovery.
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

def test_uneven_recov():
    ExperimentTypes = {"sqt_n": int, "m": int, "m_tld": int, "sigma": float, "lambda_": float,
                   "tol": float, "img": pd.CategoricalDtype(["phantom", "camera"]), 'experiment_fn_label': str, "sampling_type": pd.CategoricalDtype(["wr", "Ber"]), "emp_snr": float}
    def make_local_df():
        sqt_n = 128
        np.random.seed(0)
        df = allcombinations({'sqt_n': sqt_n, 'm': int(sqt_n**2/2), 'lambda_': 0.001, 'sigma': 0.00, 'tol': 1e-5,
                             "img": "camera", "sampling_type": ["wr", "Ber"], "adapted_sampling": True, 'save_img': True})
        # df = repeat(df, 2)
        df['seed'] = [np.random.randint(1, 2**32 - 1) for _ in range(len(df))]
        add_standard_attributes(df)
        df.attrs.update({"global_seed": 0, "level":4})
        return soft_astype_df(df, ExperimentTypes)
    local_df = make_local_df()
    local_df['seed']

    result = run_local_df_experiment(local_df, experiment, dtype=ExperimentTypes)

def test_import():
    from src import make_df, experiment
