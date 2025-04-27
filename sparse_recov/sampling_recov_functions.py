from scipy import sparse
from scipy.fft import fft, fft2, ifft, ifft2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable
import pandas as pd
from scipy.sparse.linalg import LinearOperator


def S_transpose(indices, m, m_tld, n):
    return np.sqrt(n/m) * sparse.coo_matrix(
        (np.ones(indices.shape[0]), (indices, range(m_tld))), shape=(n, m_tld))


def SF_transpose_map(indices, m, n):
    m_tld = indices.size
    S_conj = S_transpose(indices, m, m_tld, n)
    return lambda x: ifft((S_conj @ x.flat), norm="ortho")


def SF2_transpose_map(indices, m, n, shape):
    m_tld = indices.size
    S_conj = S_transpose(indices, m, m_tld, n)
    return lambda x: ifft2((S_conj @ x.flat).reshape(shape), norm="ortho")


def SDF2_transpose_map(indices, m, n, shape, d):
    m_tld = indices.size
    S_conj = S_transpose(indices, m, m_tld, n)
    return lambda x: ifft2((d*(S_conj @ x.flat)).reshape(shape), norm="ortho")


def wr_prob_vec(loc_co):
    return loc_co**2/np.sum(loc_co**2)

# def prob_weights(robust, loc_co):
#     assert num_robusts <= len(loc_co)
#     assert 0 <= num_robusts
#     p = loc_co
#     perm = np.argsort(p)
#     p = p[perm]
#     p[num_robusts:] = p[num_robusts:]**2
#     p[perm] = p
#     return p/ np.sum(p)


def Bernoulli_prob_weights(loc_co: NDArray, m: int) -> Tuple[NDArray, int, float]:
    pos_co = loc_co[loc_co > 0]
    n = pos_co.size
    assert m <= n, "Cannot sample enough measurements given entries with zero coherences cannot be sampled"
    sorted_co = np.sort(pos_co.flat)
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


def SDF2_func(indices, m, n, d):
    return lambda x: np.sqrt(n/m) * (d*fft2(x, norm="ortho").flat)[indices]


def half_robust(flat_co):
    n= len(flat_co)
    p = flat_co**2/np.sum(flat_co**2)
    p_first = p[:int(n/2)]
    tot_second = np.sum(p[int(n/2):])
    prev_tot_second = np.sum(flat_co[int(n/2):])
    if prev_tot_second==0:
        p_second= np.zeros_like(flat_co[int(n/2):])
    else:
        p_second = flat_co[int(n/2):]*tot_second/prev_tot_second 
    return np.concatenate((p_first, p_second), axis=0)


def sample_CS_matrix(loc_co, m, sampling_type, sampling_dist_type):
    d = np.array([])
    flat_co = loc_co.ravel()
    n = len(flat_co)
    if sampling_type == "wr":
        if sampling_dist_type == "robust":
            p = flat_co / np.sum(flat_co)
        elif sampling_dist_type == "half_robust":
            p = half_robust(flat_co)
        elif sampling_dist_type == "optimized":
            p = flat_co**2/np.sum(flat_co**2)
        elif sampling_dist_type == "uniform":
            p = np.repeat(1/n, n)
        else:
            raise NotImplementedError(
                "Sampling scheme " + sampling_dist_type+" not implemented")
        d = np.zeros_like(p)
        d[p > 0] = 1/np.sqrt(n * p[p > 0])
        indices = np.random.choice(n, m, replace=True, p=p)
    elif sampling_type == "Ber":
        assert m <= n, "Bernoulli sampling requires m less than n"
        if sampling_dist_type == "optimized":
            weights = Bernoulli_prob_weights(loc_co, m)[0].ravel()
            d = np.zeros_like(weights)
            d[weights > 0] = np.sqrt(m/(n*weights[weights > 0]))
        elif sampling_dist_type == "uniform":
            weights = m/n
        else:
            raise NotImplementedError()
        indices = np.where(np.random.rand(n) < weights)[0]
    else:
        raise NotImplementedError()
    return (indices, d)


def soft_threshold(x, lambda_):
    """Soft thresholding operator"""
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)


class WaveletPartialFourier(LinearOperator):
    """
    LinearOperator that implements the wavelet-partial Fourier transform.

    This class properly implements the _matvec and _rmatvec methods as required
    by the LinearOperator interface.

    Parameters:
    -----------
    idx : array-like
        Indices for the partial Fourier transform
    n : int
        Size of the wavelet coefficient vector
    matmul : Callable
        Function that implements the forward transform
    rmatmul : Callable
        Function that implements the adjoint transform
    """

    def __init__(self, idx, n, matmul: Callable, rmatmul: Callable):
        self.idx = idx
        self.shape = (len(idx), n)
        self.dtype = np.complex64
        self._matmul = matmul
        self._rmatmul = rmatmul

    def _matvec(self, x):
        """Implementation of the matrix-vector product."""
        return self._matmul(x)

    def _rmatvec(self, x):
        """Implementation of the adjoint matrix-vector product."""
        return self._rmatmul(x)
