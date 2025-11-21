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
    return lambda x: ifft2(((S_conj @ (d[indices] * x.flat))).reshape(shape), norm="ortho")


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

# Clean up to put in paper


def Optimized_Bernoulli_prob_weights(loc_co: NDArray, m: int) -> Tuple[NDArray, int, float]:
    pos_co = loc_co[loc_co > 0]
    n = pos_co.size
    assert m <= n, "Cannot sample enough measurements given entries with zero coherences cannot be sampled"

    if m == n:
        samplings = np.zeros_like(loc_co)
        samplings[loc_co > 0] = 1
        return samplings, 0, 0.0

    sorted_co = np.sort(pos_co.flat)

    def Lsqrd(j, m, n, sorted_co):
        assert j > n-m-1
        assert m <= n
        assert j <= n-1
        return m * np.sum(sorted_co[:j+1]**2) / (j-(n-m-1))

    J = None  # Is this right? It is probably never run.
    for j in range(n-1, n-m-1, -1):
        if m*sorted_co[j]**2 < Lsqrd(j, m, n, sorted_co):
            J = j
            break
    assert J != None, "Unexpected behaviour: no valid J was found."

    Lsqr = Lsqrd(J, m, n, sorted_co)

    return np.clip(m * loc_co**2 / Lsqr, 0, 1), J, Lsqr


def SF_func(indices: NDArray, m, n):
    return lambda x: np.sqrt(n/m) * fft(x, norm="ortho")[indices]


def SF2_func(indices: NDArray[np.integer], m, n):
    def return_fn(x):
        here = (np.sqrt(n/m) * fft2(x, norm="ortho")).ravel()
        return here[indices]
    return return_fn


def SDF2_func(indices, m, n, d):
    return lambda x: np.sqrt(n/m) * d[indices] * (fft2(x, norm="ortho").ravel())[indices]


def half_robust(flat_co):
    n = len(flat_co)
    p = flat_co**2/np.sum(flat_co**2)
    p_first = p[:int(n/2)]
    tot_second = np.sum(p[int(n/2):])
    prev_tot_second = np.sum(flat_co[int(n/2):])
    if prev_tot_second == 0:
        p_second = np.zeros_like(flat_co[int(n/2):])
    else:
        p_second = flat_co[int(n/2):]*tot_second/prev_tot_second
    return np.concatenate((p_first, p_second), axis=0)


def sample_wor(p, m):
    """
    Sample without replacement by re-sampling with replacement until all distinct indices are gathered.

    Args:
        p (array-like): Probabilities associated with each index (should sum to 1).

    Returns:
        tuple: (indices, counts)
            - indices: Array of unique sampled indices in order of first appearance.
            - counts: Array of the number of times each index was sampled.
    """
    n = len(p)
    indices = np.array([])
    counts = np.zeros(n, dtype=int)
    assert m >= 1
    assert len(p) >= 1

    # Maximum allowed iterations (tuned for reasonable runtime, e.g., ~1 second on typical hardware)
    MAX_ITERATIONS = int(1e7)
    # Sort probabilities in descending order
    sorted_p = np.sort(p)[::-1]
    # Estimate probability of sampling remaining indices
    if m > 1:
        # Approximate expected trials using the probability of the least likely needed index
        min_prob = sorted_p[m-1]  # Smallest probability among the m highest
        # Expected trials to hit a new index when m-1 are already sampled
        expected_trials_per_new_index = 1 / min_prob
        # Rough estimate of total trials (simplified coupon collector approximation)
        estimated_trials = m * expected_trials_per_new_index
        if estimated_trials > MAX_ITERATIONS:
            raise ValueError(
                f"Sampling {m} distinct indices with given probabilities is expected to take too long "
                f"(estimated {estimated_trials:.0f} trials). Consider reducing m or adjusting probabilities."
            )

    batch_size = max(1, int(m * 1.5))
    while True:
        samples = np.random.choice(n, size=batch_size, p=p, replace=True)
        tentative_unique = np.unique(np.concatenate([indices, samples]))
        if len(tentative_unique) > m:
            # Discard the attempt. Halve the batch size, and try again.
            batch_size = max(1, batch_size // 2)
            continue
        indices = tentative_unique
        batch_counts = np.bincount(samples, minlength=n)
        counts += batch_counts
        if len(indices) == m:
            return tentative_unique, counts
        batch_size = max(1, int((m - len(indices)) * 1.5))

```python
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def Optimized_Bernoulli_prob_weights(loc_co: NDArray, m: int) -> Tuple[NDArray, int, float]:
    pos_co = loc_co[loc_co > 0]
    n = pos_co.size
    assert m <= n, "Cannot sample enough measurements given entries with zero coherences cannot be sampled"

    if m == n:
        samplings = np.zeros_like(loc_co)
        samplings[loc_co > 0] = 1
        return samplings, 0, 0.0

    sorted_co = np.sort(pos_co.flat)

    def Lsqrd(j, m, n, sorted_co):
        assert j > n - m - 1
        assert m <= n
        assert j <= n - 1
        return m * np.sum(sorted_co[:j + 1] ** 2) / (j - (n - m - 1))

    J = None
    for j in range(n - 1, n - m - 1, -1):
        if m * sorted_co[j] ** 2 < Lsqrd(j, m, n, sorted_co):
            J = j
            break
    if J is None:
        raise ValueError("No valid J found")
    Lsqr = Lsqrd(J, m, n, sorted_co)
    return np.clip(m * loc_co ** 2 / Lsqr, 0, 1), J, Lsqr

def SF_func(indices: NDArray[np.integer], m, n):
    def return_fn(x):
        here = np.sqrt(n / m) * fft(x, norm="ortho")
        return here[indices]
    return return_fn

def SF2_func(indices: NDArray[np.integer], m, n):
    def return_fn(x):
        here = np.sqrt(n / m) * fft2(x, norm="ortho").ravel()
        return here[indices]
    return return_fn

def SDF2_func(indices, m, n, d):
    def return_fn(x):
        here = np.sqrt(n / m) * fft2(x, norm="ortho").ravel()
        return d[indices] * here[indices]
    return return_fn

def half_robust(flat_co):
    n = len(flat_co)
    sorted_idx = np.argsort(flat_co)
    sorted_co = flat_co[sorted_idx]
    p = sorted_co ** 2 / np.sum(sorted_co ** 2)
    half = int(n / 2)
    p_first = p[:half]
    tot_second = np.sum(p[half:])
    prev_tot_second = np.sum(sorted_co[half:])
    if prev_tot_second == 0:
        p_second = np.zeros_like(sorted_co[half:])
    else:
        p_second = sorted_co[half:] * tot_second / prev_tot_second
    p_sorted = np.concatenate((p_first, p_second))
    p = np.zeros_like(flat_co)
    p[sorted_idx] = p_sorted
    return p

def sample_wor(p, m):
    n = len(p)
    assert m >= 1
    assert n >= 1

    MAX_ITERATIONS = int(1e7)
    sorted_p = np.sort(p)[::-1]
    if m > 1:
        min_prob = sorted_p[m - 1]
        expected_trials_per_new_index = 1 / min_prob
        estimated_trials = m * expected_trials_per_new_index
        if estimated_trials > MAX_ITERATIONS:
            raise ValueError(f"Sampling {m} distinct indices is expected to take too long (estimated {estimated_trials:.0f} trials).")

    seen = set()
    order = []
    counts = np.zeros(n, dtype=int)
    batch_size = max(1, int(m * 1.5))
    iterations = 0

    while len(order) < m:
        if iterations > MAX_ITERATIONS:
            raise ValueError("Exceeded maximum iterations while sampling.")
        samples = np.random.choice(n, size=batch_size, p=p, replace=True)
        new_batch_counts = np.bincount(samples, minlength=n)
        tentative_new = set(samples) - seen
        if len(seen) + len(tentative_new) > m:
            batch_size = max(1, batch_size // 2)
            iterations += 1
            continue
        for s in samples:
            if s not in seen:
                seen.add(s)
                order.append(s)
        counts += new_batch_counts
        batch_size = max(1, int((m - len(order)) * 1.5))
        iterations += 1

    return np.array(order), counts

def get_sampling_dist(flat_co, m, n, sampling_dist_type, sampling_type):
    if sampling_dist_type == "uniform":
        if sampling_type in ["Ber", "adjusted_Ber", "deterministic"]:
            return np.repeat(m / n, n)
        else:
            return np.repeat(1 / n, n)
    elif sampling_dist_type == "optimized":
        if sampling_type in ["Ber", "adjusted_Ber"]:
            return Optimized_Bernoulli_prob_weights(flat_co.reshape(-1), m)[0]
        elif sampling_type == "deterministic":
            return None
        else:
            return flat_co ** 2 / np.sum(flat_co ** 2)
    elif sampling_dist_type == "robust":
        tot = np.sum(flat_co)
        if tot == 0:
            raise ValueError("Total coherence is zero.")
        return flat_co / tot
    elif sampling_dist_type == "half_robust":
        return half_robust(flat_co)
    else:
        raise NotImplementedError(f"Sampling distribution {sampling_dist_type} not implemented")

def get_precond(precond_type, p=None, weights=None, counts=None, m=None, n=None):
    if precond_type == "wr":
        if p is None:
            raise ValueError("p required for wr preconditioner")
        d = np.zeros_like(p)
      
# The problem is that precond. depends on probabilities, which may
# be either weights or wr probs. I need two different functions.


def wr_precond(p, n):
    d = np.zeros_like(p)
    d[p > 0] = 1/np.sqrt(n * p[p > 0])
    return d


def ber_precond(weights, m, n):
    d = np.zeros_like(weights)
    d[weights > 0] = np.sqrt(m/(n*weights[weights > 0]))

# depends on the optimizationdistribution.

# def d(co, preconditioner_type, ):
#     # ["wr", "ber", "emp_wor"]
#     if preconditioner == "wr":
#         p = 
#         d = np.zeros_like(p)
#         d[p > 0] = 1/np.sqrt(n * p[p > 0])
#         return d
#     elif preconditioner == "ber":
#         weights = Optimized_Bernoulli_prob_weights(co, m)[0].ravel()
#         return ber_precond(weights)
#     else:

# def get_preconditioner(probs, weights, precond_type):


def sample_CS_matrix(loc_co, m, sampling_type, sampling_dist_type, preconditioner) -> Tuple[NDArray[np.integer], NDArray]:
    d = np.array([])
    flat_co = loc_co.ravel()
    n = len(flat_co)
    p = flat_co / np.sum(flat_co)
    d = d(preconditioner)
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
    elif sampling_type == "wor":
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
        indices, counts = sample_wor(p, m)
        d = np.zeros_like(p)
        d[p > 0] = 1/np.sqrt(n * p[p > 0])
        prec_c = np.sqrt(counts)
        # normalize so that the trace is n, like I.
        normalized_c = prec_c * n/np.sum(prec_c)
        d = d * normalized_c
    elif sampling_type == "Ber":
        assert m <= n, "Bernoulli sampling requires m less than n"
        if sampling_dist_type == "optimized":
            weights = Optimized_Bernoulli_prob_weights(loc_co, m)[0].ravel()
        elif sampling_dist_type == "uniform":
            weights = np.repeat(m/n, n)
        else:
            raise NotImplementedError()
        d = np.zeros_like(weights)
        d[weights > 0] = np.sqrt(m/(n*weights[weights > 0]))
        indices = np.where(np.random.rand(n) < weights)[0]
    elif sampling_type == "adjusted_Ber":
        assert m <= n, "Bernoulli sampling requires m (" + str(
            m) + ") less than n (" + str(n) + ")."
        if sampling_dist_type == "optimized":
            weights = Optimized_Bernoulli_prob_weights(loc_co, m)[0].ravel()
        elif sampling_dist_type == "uniform":
            weights = np.repeat(m/n, n)
        else:
            raise NotImplementedError()
        d = np.zeros_like(weights)
        d[weights > 0] = np.sqrt(m/(n*weights[weights > 0]))
        MAX_NUMTRIES = int(1e4)
        indices = None
        for i in range(MAX_NUMTRIES):
            indices = np.where(np.random.rand(n) < weights)[0]
            if len(indices) == m:
                break
        if indices is None:
            raise ValueError("could not sample conditional Bernoulli")
    elif sampling_type == "deterministic":
        assert m <= n, "Bernoulli sampling requires m less than n"
        weights = np.zeros_like(flat_co)
        # Get indices of m largest entries
        top_m_indices = np.argsort(flat_co)[-m:]
        weights[top_m_indices] = 1.0
        d = np.zeros_like(weights)
        d[weights > 0] = np.sqrt(m / n)
        indices = np.where(np.random.rand(n) < weights)[0]
    else:
        raise NotImplementedError()

    if preconditioner is not None:
        if preconditioner =

    return (indices.astype(int), d)


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
