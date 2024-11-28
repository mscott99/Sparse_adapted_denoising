from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import datetime
import numpy as np
from typing import TypedDict, NotRequired, Literal, Any, Tuple
from skimage.data import shepp_logan_phantom, camera
from skimage import transform
from scipy import sparse
from pywt import dwt2, idwt2, wavedec2, waverec2, ravel_coeffs, unravel_coeffs
from utils import logspace, allcombinations, add_standard_attributes, run_local_df_experiment, repeat, soft_astype_df
from scipy.fft import fft, ifft


def SF_transpose_map(indices, m, n):
    m_tld = indices.size
    S_conj = S_transpose(indices, m_tld, n)
    return lambda x: np.sqrt(n/m) * ifft((S_conj @ x.flat), norm="ortho")


def wr_prob_vec(loc_co):
    return loc_co**2/np.sum(loc_co**2)


def Bernoulli_prob_weights(loc_co: NDArray, m: int) -> Tuple[NDArray, int, int]:
    assert (loc_co > 0).all()
    n = loc_co.size
    assert m < n
    sorted_co = np.sort(loc_co)
    J = n

    def Lsqrd(j, m, n, sorted_co):
        return m * np.sum(sorted_co[:j]**2) / (m - (n-j))
    for j in range(n, n-m-1, -1):
        if m*sorted_co[j-1]**2 < Lsqrd(j, m, n, sorted_co):
            J = j
            break
    Lsqr = Lsqrd(J, m, n, sorted_co)
    return np.clip(m * loc_co**2 / Lsqr, 0, 1), J, Lsqr


def SF_func(indices: NDArray, m, n):
    return lambda x: np.sqrt(n/m) * fft(x, norm="ortho").flat[indices]


def S_transpose(indices, m_tld, n):
    S = sparse.coo_matrix(
        (np.ones(indices.shape[0]), (indices, range(m_tld))), shape=(n, m_tld))
    return S


def soft_threshold(x, lambda_):
    """Soft thresholding operator"""
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)


def lasso_wr_Fourier(n, m, indices, b: np.ndarray, x: np.ndarray, lambda_, d=None, max_iter=1000, tol=1e-5):
    """LASSO solver using Iterative Soft Thresholding Algorithm (ISTA)
    If no resampling, set L to 2
    """
    assert len(indices) > 0
    # L is the Lipshitz constant of the gradient, which scales with the operator norm of S^TS, which is the max number of repeated measurements.
    L = 2 * np.max(np.unique(indices, return_counts=True)[1]) * n/m
    if d is None:
        d = np.ones(n)
    d_tld = d[indices]
    x_old = np.zeros_like(x)
    SF = SF_func(indices, m, n)
    SF_transpose = SF_transpose_map(indices, m, n)
    fits = np.zeros(max_iter)
    regs = np.zeros(max_iter)
    for ind in range(max_iter):
        x_old[:] = x
        residual = (b - SF(x))
        fits[ind] = np.linalg.norm(residual)**2
        regs[ind] = np.sum(np.abs(x))
        gradient = -2*np.real(SF_transpose(d_tld**2*residual))
        x = soft_threshold(x - gradient / L, lambda_ / L)
        if np.linalg.norm(x - x_old) < tol:
            break
    return x, fits, regs


ExperimentTypes = {"sqt_n": int, "m": int, "m_tld": int, "sigma": float, "lambda_": float,
                   "tol": float, "img": pd.CategoricalDtype(["phantom", "camera"]), 'experiment_fn_label': str, "sampling_type": pd.CategoricalDtype(["wr", "Ber"]), "emp_snr": float}


def make_df():
    sqt_n = 400
    df = allcombinations({'sqt_n': sqt_n, 'm': logspace(1e2, sqt_n**2, 8, integer=True), 'sigma': [0.0, 0.001, 0.01, 0.1, 10.0], 'lambda_': [
        0.01, 0.001, 0.0001], 'tol': 1e-5, 'img': ["phantom", "camera"], "sampling_type": ["wr", "Ber"], "adapted_sampling": True})
    df = repeat(df, 3)
    df['img'] = df['img'].astype('category')
    global_seed = 1234
    np.random.seed(global_seed)
    df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
    add_standard_attributes(df)
    df.attrs.update(
        {"label": "fixed", "global_seed": global_seed, "save_img": False})
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
    sampling_type: Literal["wr", "Ber", "wor"]
    adapted_sampling: bool
    loc_co: NotRequired[np.ndarray]
    global_seed: NotRequired[int]
    save_img: bool
    m_tld: NotRequired[int]
    recovered_signal: NotRequired[NDArray[np.float32]]
    datetime: NotRequired[datetime.datetime]
    emp_snr: NotRequired[np.float32]
    commit: NotRequired[str]
    rrep: NotRequired[float]
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
    # features = square_haar(dwt2(true_signal, "haar"))
    features, coeff_slices, coeff_shape = ravel_coeffs(
        wavedec2(true_signal, "haar", level=5))
    # Override n because the wavelet transform may change it slightly.
    n = features.size
    # The sparse basis is fully incoherent with the Fourier basis in this experiment.
    data['loc_co'] = np.repeat(1/np.sqrt(n), n)
    np.random.seed(data['seed'])
    d = None
    if data['sampling_type'] == "wr":
        if data['adapted_sampling'] == True:
            assert 'loc_co' in data
            p = data['loc_co']**2/np.sum(data['loc_co']**2)
            indices = np.random.choice(
                n, m, replace=True, p=p)
            d = 1/np.sqrt(n * p)
        else:
            indices = np.random.choice(n, m, replace=True)
    elif data['sampling_type'] == "Ber":
        assert m <= n
        if data['adapted_sampling']:
            assert "loc_co" in data
            weights = Bernoulli_prob_weights(data['loc_co'], m)[0]
            d = np.sqrt(m/(n*weights))
        else:
            weights = m/n
        indices = np.where(np.random.rand(n) < weights)[0]
    else:
        raise NotImplementedError()
    SF = SF_func(indices, m, n)
    m_tld = indices.size
    unit_features = features/np.linalg.norm(features)
    signal = SF(unit_features)
    noise = 1/np.sqrt(m) * data['sigma'] * np.random.randn(m_tld)
    b = signal + noise
    if data['sigma'] == 0:
        emp_snr = np.inf
    else:
        emp_snr = np.linalg.norm(signal)/np.linalg.norm(noise)
    x_hat, fits, regs = lasso_wr_Fourier(n, m, indices, b, np.zeros_like(
        unit_features), data['lambda_'], tol=data['tol'], d=d)
    rre = np.linalg.norm(x_hat - unit_features)
    run_time = time()-start_time
    return_dict = {'rrep': rre, 'runtime_seconds': run_time, 'fits': fits,
                   'regs': regs, 'indices': indices, 'experiment_fn_label': "fully_sparse_exp", "m_tld": m_tld, 'emp_snr': emp_snr}
    if data['save_img']:
        return_dict['recovered_signal'] = waverec2(unravel_coeffs(
            x_hat, coeff_slices, coeff_shape, output_format="wavedec2"), "haar")
    return return_dict


def make_local_df():
    sqt_n = 100
    np.random.seed(0)
    df = allcombinations({'sqt_n': sqt_n, 'm': int(sqt_n**2/2), 'lambda_': 0.001, 'sigma': 0.00, 'tol': 1e-5,
                         "img": "camera", "sampling_type": ["wr", "Ber"], "adapted_sampling": [True, True, False], 'save_img': True})
    # df = repeat(df, 2)
    df['seed'] = [np.random.randint(1, 2**32 - 1) for _ in range(len(df))]
    add_standard_attributes(df)
    df.attrs.update({"global_seed": 0})
    return soft_astype_df(df, ExperimentTypes)

# local_df = make_local_df()
# local_df['seed']
# result = run_local_df_experiment(local_df, experiment, dtype=ExperimentTypes)
# result.to_pickle("./data/local_larger_12.pickle")

# print(result['TRACEBACK'][0])
# plt.imshow(result["recovered_signal"][0])
# soft_astype_df(result, ExperimentTypes)
# result['runtime_seconds']

# plt.figure()
# plt.plot(result['fits'][0])
# plt.plot(result['regs'][0])
# plt.yscale('log')
# np.max(result['rre'])
