from scipy import sparse
from scipy.fft import fft, ifft, ifft2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def S_transpose(indices, m_tld, n):
    return sparse.coo_matrix(
        (np.ones(indices.shape[0]), (indices, range(m_tld))), shape=(n, m_tld))


def SF_transpose_map(indices, m, n):
    m_tld = indices.size
    S_conj = S_transpose(indices, m_tld, n)
    return lambda x: np.sqrt(n/m) * ifft((S_conj @ x.flat), norm="ortho")


def SF2_transpose_map(indices, m, n, shape):
    m_tld = indices.size
    S_conj = S_transpose(indices, m_tld, n)
    return lambda x: np.sqrt(n/m) * ifft2((S_conj @ x.flat).reshape(shape), norm="ortho")


def wr_prob_vec(loc_co):
    return loc_co**2/np.sum(loc_co**2)


def Bernoulli_prob_weights(loc_co: NDArray, m: int) -> Tuple[NDArray, int, float]:
    assert (loc_co > 0).all()
    n = loc_co.size
    assert m < n
    sorted_co = np.sort(loc_co.flat)
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
    return lambda x: np.sqrt(n/m) * fft(x, norm="ortho")[indices]


def SF2_func(indices: NDArray, m, n):
    return lambda x: np.sqrt(n/m) * fft2(x, norm="ortho").flat[indices]


def soft_threshold(x, lambda_):
    """Soft thresholding operator"""
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
