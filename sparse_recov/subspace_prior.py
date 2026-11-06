import pandas as pd
from time import time
import scipy
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
from typing import TypedDict, Literal, Any, Tuple, List
from skimage.data import shepp_logan_phantom, camera
from skimage import transform
from utils.utils_data import logspace, allcombinations, add_standard_attributes, run_local_df_experiment, repeat, soft_astype_df
from utils.utils_plot import show_vec_mag
from scipy.fft import fftshift, fft2

ExperimentTypes = {"sqt_n": int, "k": int, "m": int, "m_tld": int, "sigma": float, "level": int, "img": pd.CategoricalDtype(
    ["phantom", "camera"]), 'experiment_fn_label': str, "sampling_type": pd.CategoricalDtype(["wr", "Ber"]), "emp_snr": float, "preconditioned_optimization": bool}

def make_df():
    sqt_n = 256
    df = allcombinations({'m': logspace(1e2, sqt_n**2*4, 6, integer=True), 'k': [100], 'sigma': [
        0.0, 1e-4, 1e-2, 1e0], 'max_iter': 1e6, 'img': "camera", "sampling_type": "wr", "adapted_sampling": True, "preconditioned_optimization": [True, False]})
    # df = repeat(df, 2)
    df['img'] = df['img'].astype('category')
    global_seed = 1234
    np.random.seed(global_seed)
    df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
    add_standard_attributes(df)
    df.attrs.update(
        {'sqt_n': sqt_n, "label": "uneven_sparser", "global_seed": global_seed, "save_img": False, "level": 5, "save_sampling": True})
    df = soft_astype_df(df, ExperimentTypes)
    return df

# def experiment(data: dict) -> dict[str, Any]:
#     np.random.seed(data['seed'])
#     start_time = time()
#     sqt_n = int(data['sqt_n'])
#     n = sqt_n**2
#
#     # Generate the true signal
#     if data['img'] == "phantom":
#         true_signal = transform.resize(
#             shepp_logan_phantom().astype(np.float32), (sqt_n, sqt_n))
#     elif data['img'] == "camera":
#         true_signal = transform.resize(
#             camera().astype(np.float32), (sqt_n, sqt_n))
#     else:
#         raise NotImplementedError()
#
#     Q, _ = scipy.linalg.qr(np.random.randn(data['m'], n))
#     # Project the true signal on the range of prior mat. Use the projection operator, from the transpose multiplied with the prior_mat. Give me only the line of code.
#
#     Q.T
#
# Claude, write an experiment() function for compressed sensing recovery similarly to the sparse case, but with a linear prior, meaning that there is no longer any sparsity involved here, only subsampled fft in the same way as above, and a linear prior set. It should be the range of Q.T. The optimization recovery should use a least squares solver. The local coherences should be found in its own function, taking the alignment of Fourier vectors (dot product) with the range of Q.T.

def compute_local_coherences(sqt_n: int, Q: NDArray) -> NDArray:
    """Compute local coherences between Fourier basis and range of Q.T"""
    n = sqt_n**2
    local_coherences = np.zeros((sqt_n, sqt_n))
    
    # Get orthonormal basis for range of Q.T
    V = scipy.linalg.orth(Q.T)  # This is more stable than using Q.T directly
    
    # For each Fourier basis vector
    for i in range(sqt_n):
        for j in range(sqt_n):
            # Create Fourier basis vector
            f = np.zeros((sqt_n, sqt_n), dtype=np.complex128)
            f[i,j] = 1
            f_vec = fft2(f, norm="ortho").ravel()
            
            # Compute projection norm onto range of Q.T
            proj_norm = np.linalg.norm(V @ (V.T.conj() @ f_vec))
            local_coherences[i,j] = proj_norm

    return local_coherences

# below fn written by ai go through with debugger.
def experiment(data: dict) -> dict[str, Any]:
    np.random.seed(data['seed'])
    start_time = time()
    sqt_n = int(data['sqt_n'])
    n = sqt_n**2

    # Generate the true signal
    if data['img'] == "phantom":
        true_signal = transform.resize(
            shepp_logan_phantom().astype(np.float32), (sqt_n, sqt_n))
    elif data['img'] == "camera":
        true_signal = transform.resize(
            camera().astype(np.float32), (sqt_n, sqt_n))
    else:
        raise NotImplementedError()
    
    # Generate random prior matrix and get its range
    V = scipy.linalg.orth(np.random.randn(data['m'], n).T)  # Orthonormal basis for range of Q.T
    
    # Project true signal onto range of Q.T
    true_signal_flat = true_signal.ravel()
    true_signal_flat = V @ (V.T.conj() @ true_signal_flat)
    true_signal = true_signal_flat.reshape((sqt_n, sqt_n))
    true_signal = true_signal/np.linalg.norm(true_signal)

    # Compute local coherences for sampling
    loc_co = compute_local_coherences(sqt_n, Q)
    
    # Sampling
    m = data['m']
    d = np.array([])
    if data['sampling_type'] == "wr":
        if data['adapted_sampling'] == True:
            p = (loc_co**2/np.sum(loc_co**2)).ravel()
            indices = np.random.choice(n, m, replace=True, p=p)
            d = np.zeros_like(p)
            d[p > 0] = 1/np.sqrt(n * p[p > 0])
        else:
            indices = np.random.choice(n, m, replace=True)
    elif data['sampling_type'] == "Ber":
        assert m <= n
        if data['adapted_sampling']:
            weights = Bernoulli_prob_weights(loc_co, m)[0].ravel()
            d = np.zeros_like(weights)
            d[weights > 0] = np.sqrt(m/(n*weights[weights > 0]))
        else:
            weights = m/n
        indices = np.where(np.random.rand(n) < weights)[0]
    else:
        raise NotImplementedError()

    m_tld = indices.size
    
    # Measurement operator
    SF = SF2_func(indices, m, n)
    signal = SF(true_signal)
    noise = 1/np.sqrt(m) * data['sigma'] * np.random.randn(m_tld)
    b = signal + noise

    # Recovery via least squares
    if data['preconditioned_optimization']:
        if data['sampling_type'] == 'Ber':
            raise NotImplementedError()
        SDF = SDF2_func(indices, m, n, d)
        A = lambda x: SDF(x.reshape(sqt_n, sqt_n)).ravel()
        AT = lambda x: SDF2_transpose_map(indices, m, n, (sqt_n, sqt_n), d)(x)
        b_scaled = d[indices]*b
    else:
        A = lambda x: SF(x.reshape(sqt_n, sqt_n)).ravel()
        AT = lambda x: SF2_transpose_map(indices, m, n, (sqt_n, sqt_n))(x)
        b_scaled = b

    # Form the normal equations matrix for the least squares problem
    def matvec(x):
        return V @ (V.T.conj() @ AT(A(V @ x)))
    
    linear_op = scipy.sparse.linalg.LinearOperator((V.shape[1], V.shape[1]), matvec=matvec)
    rhs = V.T.conj() @ AT(b_scaled)
    
    # Solve the normal equations
    coeffs, info = scipy.sparse.linalg.cg(linear_op, rhs)
    x_hat = (V @ coeffs).reshape(sqt_n, sqt_n)

    # Compute metrics and return
    emp_snr = np.inf if data['sigma'] == 0 else np.linalg.norm(signal)/np.linalg.norm(noise)
    re = np.linalg.norm(x_hat - true_signal)/np.linalg.norm(true_signal)
    run_time = time()-start_time
    
    return_dict = {
        're': re, 
        'runtime_seconds': run_time,
        'indices': indices,
        'experiment_fn_label': "linear_prior",
        'm_tld': m_tld,
        'emp_snr': emp_snr,
        'noise_mag': np.linalg.norm(noise),
        'signal_mag': np.linalg.norm(signal),
        'optim_info': info
    }
    
    if 'save_img' in data and data['save_img']:
        return_dict['recovered_signal'] = x_hat
    if 'save_sampling' in data and data['save_sampling']:
        return_dict['indices'] = indices
        
    return return_dict



    #
    # # Override n because the wavelet transform may change it slightly.
    # m = data['m']
    # n = true_signal.size
    # loc_co = Haar_local_coherence(sqt_n, data['level'])
    #
    # d = np.array([])
    # if data['sampling_type'] == "wr":
    #     if data['adapted_sampling'] == True:
    #         p = (loc_co**2/np.sum(loc_co**2)).ravel()
    #         indices = np.random.choice(
    #             n, m, replace=True, p=p)
    #         d = np.zeros_like(p)
    #         d[p > 0] = 1/np.sqrt(n * p[p > 0])
    #     else:
    #         indices = np.random.choice(n, m, replace=True)
    # elif data['sampling_type'] == "Ber":
    #     assert m <= n
    #     if data['adapted_sampling']:
    #         weights = Bernoulli_prob_weights(loc_co, m)[0].ravel()
    #         d = np.zeros_like(weights)
    #         d[weights > 0] = np.sqrt(m/(n*weights[weights > 0]))
    #     else:
    #         weights = m/n
    #     indices = np.where(np.random.rand(n) < weights)[0]
    # else:
    #     raise NotImplementedError()
    #
    # m_tld = indices.size
    # SF = SF2_func(indices, m, n)
    #
    # # Signal acquisition
    # signal = SF(true_signal)
    # noise = 1/np.sqrt(m) * data['sigma'] * np.random.randn(m_tld)
    # b = signal + noise
    #
    # if data['preconditioned_optimization']:
    #     if data['sampling_type'] == 'Ber':
    #         raise NotImplementedError(
    #             "Bernoulli sampling with preconditioned opt not implemented")
    #     SDF = SDF2_func(indices, m, n, d)
    #     SDF_transpose = SDF2_transpose_map(indices, m, n, true_signal.shape, d)
    #     A = WaveletPartialFourier(indices, np.prod(theshape), lambda x: SDF(
    #         inverse_wavelet(x)), lambda x: wavelet_transform(SDF_transpose(x)))
    #     b = d[indices]*b
    # else:
    #     SF_transpose = SF2_transpose_map(indices, m, n, true_signal.shape)
    #     A = WaveletPartialFourier(indices, np.prod(theshape), lambda x: SF(
    #         inverse_wavelet(x)), lambda x: wavelet_transform(SF_transpose(x)))
    #
    # # Recover the signal
    # lambda_ = data['lambda_coeff']*data['sigma']/np.sqrt(m) + 1e-8
    # x_haar, resid, grad, opt_info = spgl1.spg_bpdn(
    #     A, b, lambda_, iter_lim=data['max_iter'], verbosity=0)
    # x_hat = inverse_wavelet(np.real(x_haar))
    #
    # # Return relevant data.
    # if data['sigma'] == 0:
    #     emp_snr = np.inf
    # else:
    #     emp_snr = np.linalg.norm(signal)/np.linalg.norm(noise)
    # # optim_info = {'resid': resid, 'grad': grad, 'info': info}
    # re = np.linalg.norm(x_hat - true_signal)/np.linalg.norm(true_signal)
    # run_time = time()-start_time
    # return_dict = {'re': re, 'runtime_seconds': run_time, 'indices': indices, 'experiment_fn_label': "uneven_on_exactly_sparse",
    #                "m_tld": m_tld, 'emp_snr': emp_snr, 'noise_mag': np.linalg.norm(noise), 'signal_mag': np.linalg.norm(signal), 'optim_info': opt_info}
    # if 'save_img' in data and data['save_img']:
    #     return_dict['recovered_signal'] = x_hat
    # if 'save_sampling' in data and data['save_sampling']:
    #     return_dict['indices'] = indices
    # return return_dict
