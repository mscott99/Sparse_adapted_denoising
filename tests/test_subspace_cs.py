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

from src.uneven_sparse_experiment import experiment,  make_df

def test_experiment():
    df = make_df()
    experiment({**df.attrs, **df.loc[0]})  # test for runtime errors
