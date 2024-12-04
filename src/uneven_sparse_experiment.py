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

from .sampling_recov_functions import SF2_func, SF2_transpose_map, soft_threshold, Bernoulli_prob_weights


def get_representative_points(sqt_n: int, level: int = 1) -> List[Tuple]:
    points = [(0, 0)]
    for i in range(level):
        points.append((0, int(np.ceil(sqt_n/2**(i+1)))))
        points.append((int(np.ceil(sqt_n/2**(i+1))),
                      int(np.ceil(sqt_n/2**(i+1)))))
        points.append((int(np.ceil(sqt_n/2**(i+1))), 0))
    return points


def Haar_local_coherence(sqt_n: int, level: int):
    img_to_reconstruct, slices = coeffs_to_array(
        wavedec2(np.zeros((sqt_n, sqt_n)), "haar", level=level))
    max_co = np.zeros((sqt_n, sqt_n))
    for i, j in get_representative_points(sqt_n, level):
        img_to_reconstruct.fill(0)
        img_to_reconstruct[i, j] = 1
        new_co = np.abs(fft2(waverec2(array_to_coeffs(
            img_to_reconstruct, slices, output_format="wavedec2"), "haar"), norm="ortho"))
        max_co = np.maximum(new_co, max_co)
    return max_co


# sqt_n = 512
# level = 7
# max_co = Haar_local_coherence(sqt_n, level)

# For more depth, put one point in each category.

# plt.figure()
# sns.heatmap(fftshift(max_co), norm=LogNorm())

# features, coeff_slices, coeff_shape = ravel_coeffs(
#     wavedec2(np.random.rand(2, 2), "haar", level=5))  # Padding all around
# features.shape
# e1 = np.zeros_like(coeffs)
# e1[10, 0] = 1
# reconstruct = waverec2(array_to_coeffs(
#     e1, slices, output_format="wavedec2"), "haar")
# reconstruct
# np.abs(fft2(reconstruct))

def lasso_Fourier_haar_basis(n: int, m: int, indices: NDArray, b: np.ndarray, x: np.ndarray, lambda_: float, wavelet="haar", level=5, d=None, max_iter=1000, tol=1e-5):
    assert len(indices) > 0
    # L is the Lipshitz constant of the gradient, which scales with the operator norm of S^TS, which is the max number of repeated measurements.
    if d is None:
        d = np.ones(n)
    d_tld = d[indices]
    L = 2 * np.max(np.unique(indices, return_counts=True)[1]) * n/m* np.max(d_tld)**2
    # To go fast, I would need to check the coherence at each iterate, to increase step size
    haar_x, slices = coeffs_to_array(wavedec2(x, wavelet, level=level))
    def wavelet_transform(y: np.ndarray) -> np.ndarray:
        return coeffs_to_array(wavedec2(y, wavelet, level=level))[0]
    def inverse_wavelet(w: np.ndarray) -> np.ndarray:
        return waverec2(array_to_coeffs(
            w, slices, output_format="wavedec2"), wavelet)
    old_wavelet = np.zeros_like(haar_x)
    SF = SF2_func(indices, m, n)
    SF_transpose = SF2_transpose_map(indices, m, n, x.shape)
    fits = np.zeros(max_iter)
    regs = np.zeros(max_iter)
    for ind in range(max_iter):
        old_wavelet[:] = haar_x
        residual = d_tld*(b - SF(inverse_wavelet(haar_x)))
        fits[ind] = np.linalg.norm(residual)**2
        regs[ind] = np.sum(np.abs(haar_x))
        gradient = -2 * \
            wavelet_transform(np.real(SF_transpose(d_tld*residual)))
        haar_x = soft_threshold(haar_x - gradient / L, lambda_ / L)
        if np.linalg.norm(haar_x - old_wavelet) < tol:
            break
    return inverse_wavelet(haar_x), fits, regs


ExperimentTypes = {"sqt_n": int, "m": int, "m_tld": int, "sigma": float, "lambda_": float,
                   "tol": float, "level": int, "img": pd.CategoricalDtype(["phantom", "camera"]), 'experiment_fn_label': str, "sampling_type": pd.CategoricalDtype(["wr", "Ber"]), "emp_snr": float}


def make_df():
    sqt_n = 400
    df = allcombinations({'sqt_n': sqt_n, 'm': logspace(1e2, sqt_n**2, 8, integer=True), 'sigma': [0, 0.01, 0.1, 1.0], 'lambda_': [
        0.01, 0.001, 0.0001], 'tol': 1e-6, 'img': ["phantom", "camera"], "sampling_type": ["wr", "Ber"], "adapted_sampling": [True, False]})
    df = repeat(df, 3)
    df['img'] = df['img'].astype('category')
    global_seed = 1234
    np.random.seed(global_seed)
    df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
    add_standard_attributes(df)
    df.attrs.update(
        {"label": "fixed", "global_seed": global_seed, "save_img": False, "level": 5, "save_sampling": True})
    df = soft_astype_df(df, ExperimentTypes)
    return df


class ExperimentData(TypedDict):
    sqt_n: int
    m: int
    sigma: float
    lambda_: float
    tol: float
    img: str
    seed: int
    save_sampling: bool
    sampling_type: Literal["wr", "Ber", "wor"]
    adapted_sampling: bool
    loc_co: NotRequired[np.ndarray]
    global_seed: NotRequired[int]
    save_img: bool
    level: int
    m_tld: NotRequired[int]
    recovered_signal: NotRequired[NDArray[np.float32]]
    datetime: NotRequired[datetime.datetime]
    emp_snr: NotRequired[np.float32]
    commit: NotRequired[str]
    re: NotRequired[float]
    runtime_seconds: NotRequired[int]
    fits: NotRequired[np.ndarray]
    regs: NotRequired[np.ndarray]
    indices: NotRequired[list]
    experiment_fn_label: NotRequired[str]


def experiment(data: ExperimentData) -> dict[str, Any]:
    np.random.seed(data['seed'])
    start_time = time()
    sqt_n = int(data['sqt_n'])
    n = sqt_n**2
    m = data['m']
    if data['img'] == "phantom":
        true_signal = transform.resize(
            shepp_logan_phantom().astype(np.float32), (sqt_n, sqt_n))
    elif data['img'] == "camera":
        true_signal = transform.resize(
            camera().astype(np.float32), (sqt_n, sqt_n))
    else:
        raise NotImplemented()
    # Override n because the wavelet transform may change it slightly.
    n = true_signal.size
    loc_co = Haar_local_coherence(sqt_n, data['level'])
    np.random.seed(data['seed'])
    d = None
    if data['sampling_type'] == "wr":
        if data['adapted_sampling'] == True:
            p = (loc_co**2/np.sum(loc_co**2)).ravel()
            indices = np.random.choice(
                n, m, replace=True, p=p)
            d = 1/np.sqrt(n * p)
        else:
            indices = np.random.choice(n, m, replace=True)
    elif data['sampling_type'] == "Ber":
        assert m <= n
        if data['adapted_sampling']:
            weights = Bernoulli_prob_weights(loc_co, m)[0].ravel()
            d = np.zeros_like(weights)
            d[weights>0] = np.sqrt(m/(n*weights[weights>0]))
        else:
            weights = m/n
        indices = np.where(np.random.rand(n) < weights)[0]
    else:
        raise NotImplementedError()
    SF = SF2_func(indices, m, n)
    m_tld = indices.size
    norm_const = np.linalg.norm(true_signal)
    unit_features = true_signal/norm_const
    signal = SF(unit_features)
    noise = 1/np.sqrt(m) * data['sigma'] * np.random.randn(m_tld)
    b = signal + noise
    if data['sigma'] == 0:
        emp_snr = np.inf
    else:
        emp_snr = np.linalg.norm(signal)/np.linalg.norm(noise)
    x_hat, fits, regs = lasso_Fourier_haar_basis(n, m, indices, b, np.zeros_like(
        unit_features), data['lambda_'], tol=data['tol'], d=d)
    re = np.linalg.norm(x_hat - unit_features)
    run_time = time()-start_time
    return_dict = {'re': re, 'runtime_seconds': run_time, 'fits': fits,
                   'regs': regs, 'indices': indices, 'experiment_fn_label': "fully_sparse_exp", "m_tld": m_tld, 'emp_snr': emp_snr}
    if data['save_img']:
        return_dict['recovered_signal'] = x_hat * norm_const
    if data['save_sampling']:
        return_dict['indices'] = indices
    return return_dict


def make_local_df():
    sqt_n = 400
    df = allcombinations({'sqt_n': sqt_n, 'm': 40000, 'sigma': 0, 'lambda_':  0.0001, 'tol': 1e-6, 'img': ["phantom", "camera"], "sampling_type":"Ber", "adapted_sampling": True})
    df = repeat(df, 1)
    df['img'] = df['img'].astype('category')
    global_seed = 1234
    np.random.seed(global_seed)
    df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
    add_standard_attributes(df)
    df.attrs.update(
        {"label": "fixed", "global_seed": global_seed, "save_img": True, "level": 5, "save_sampling": True})
    df = soft_astype_df(df, ExperimentTypes)
    return df

# local_df = make_local_df()

# result = run_local_df_experiment(local_df, experiment, dtype=ExperimentTypes)
# plt.figure()
# plt.heatmap(Haar_local_coherence(400, 5), norm=LogNorm())
# sns.heatmap(Haar_local_coherence(400, 5), norm=LogNorm())

# result['re'][0]

# loc_co = Haar_local_coherence(400, 5)

# non_zero_mask = loc_co > 0
# pos_co = loc_co[non_zero_mask]
# n = pos_co.size
# loc_co.size

# plt.imshow(result['recovered_signal'][1])

# plt.imshow(result['m_tld'])
#
# result['fits']
#
# plt.yscale("log")
#
# plt.figure()
# plt.plot(range(len(result['fits'][0])),result['fits'][0])
# plt.plot(range(len(result['fits'][0])),result['regs'][0])
#
# np.max(result.regs[0])
#
# result['regs'][0][0]
# result['fits'][0][0]


# print(result['TRACEBACK'][0])
# plt.imshow(result["recovered_signal"][0])
# soft_astype_df(result, ExperimentTypes)
# result['runtime_seconds']

# plt.figure()
# plt.plot(result['fits'][0])
# plt.plot(result['regs'][0])
# plt.yscale('log')
# np.max(result['rre'])
