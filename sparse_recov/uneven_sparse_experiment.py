# This file is for uneven signal recovery.
from utils.utils_data import allcombinations, cross_merge_dfs, repeat, run_local_df_experiment
import os
import torchvision
from torchvision import transforms
import pandas as pd
from time import time
import numpy as np
from typing import Any, Tuple, List
from skimage.data import shepp_logan_phantom, camera
from skimage import transform
from pywt import wavedec2, waverec2, coeffs_to_array, array_to_coeffs
from utils.utils_data import logspace, allcombinations, add_standard_attributes, repeat, run_local_df_experiment, soft_astype_df
from scipy.fft import fft2
import spgl1

from sparse_recov.sampling_recov_functions import SF2_func, SDF2_func, SF2_transpose_map, SDF2_transpose_map, soft_threshold, Bernoulli_prob_weights, WaveletPartialFourier, sample_CS_matrix


def get_representative_points(sqt_n: int, level: int = 1) -> List[Tuple]:
    points = [(0, 0)]
    for i in range(level):
        points.append((0, int(np.ceil(sqt_n/2**(i+1)))))
        points.append((int(np.ceil(sqt_n/2**(i+1))),
                      int(np.ceil(sqt_n/2**(i+1)))))
        points.append((int(np.ceil(sqt_n/2**(i+1))), 0))
    return points


def Haar_local_coherence(signal_shape, level: int):
    assert signal_shape[1] == signal_shape[2]
    axes = (len(signal_shape)-2, len(signal_shape)-1)
    sqt_n = signal_shape[1]
    img_to_reconstruct, slices = coeffs_to_array(
        wavedec2(np.zeros(signal_shape), "haar", level=level), axes=axes)
    max_co = np.zeros(signal_shape)
    for i, j in get_representative_points(sqt_n, level):
        for k in range(img_to_reconstruct.shape[0]):
            img_to_reconstruct.fill(0)
            img_to_reconstruct[k, i, j] = 1
            this_vec = waverec2(array_to_coeffs(
                img_to_reconstruct, slices, output_format="wavedec2"), "haar")
            normalized_vec = this_vec/np.sqrt(np.sum(this_vec**2))
            new_co = np.abs(fft2(normalized_vec, norm="ortho"))
            max_co = np.maximum(new_co, max_co)
    return max_co


ExperimentTypes = {"sqt_n": int, "k": int, "m": int, "m_tld": int, "sigma": float, "lambda_coeff": float, "level": int, "img": pd.CategoricalDtype(
    ["celeba", "phantom", "camera"]), 'experiment_fn_label': str, "sampling_type": pd.CategoricalDtype(["wor", "adjusted_Ber", "wr", "Ber", "deterministic"]), "emp_snr": float, "preconditioned_optimization": bool, "sampling_dist_type": str, "sparsity": pd.CategoricalDtype(["prop_random", "optimized", "top"])}

# Using the functions above, make a function that takes a 2d ndarray, transforms it,
# and truncates its sparsity level in the Haar basis to a specified sparsity level.


def threshold_vec(v, k):
    sorted_idx = np.argsort(np.abs(v.ravel()))
    threshold_idx = sorted_idx[-k:]
    sparse_coeffs = np.zeros_like(v)
    sparse_coeffs.ravel()[threshold_idx] = v.flatten()[threshold_idx]
    return sparse_coeffs


def rand_threshold_vec(v, k, p=None):
    """
    Sparsify a vector by randomly selecting k indices to keep, with optional weights.

    Args:
        v: Input vector (numpy array)
        k: Number of non-zero elements to keep
        p: Probability weights for sampling (optional, if None uniform weights are used)

    Returns:
        Sparsified vector with k non-zero elements
    """
    n = v.size
    if p is None:
        p = np.ones(n) / n
    assert k <= n
    selected_idx = np.random.choice(n, size=k, replace=False, p=p/np.sum(p))
    sparse_coeffs = np.zeros_like(v)
    sparse_coeffs.ravel()[selected_idx] = v.flatten()[selected_idx]
    return sparse_coeffs


def truncate_to_sparsity(image: np.ndarray, k: int, to_wavelet, from_wavelet, truncation_type: str = "top") -> np.ndarray:
    """
    Truncate an image to a specified sparsity level in the Haar wavelet basis.

    Args:
        image: 2D input array
        sparsity_level: Number of coefficients to keep
        level: Wavelet decomposition level
        wavelet: Wavelet type (default: 'haar')

    Returns:
        Reconstructed image with specified sparsity level
    """

    # Transform to wavelet domain
    if truncation_type == "top":
        return from_wavelet(threshold_vec(to_wavelet(image), k))
    elif truncation_type == "prop_random":
        coeffs = to_wavelet(image)
        return from_wavelet(rand_threshold_vec(coeffs, k, np.abs(coeffs)))
    elif truncation_type == "optimized":
        coeffs = to_wavelet(image)
        return from_wavelet(rand_threshold_vec(coeffs, k, coeffs**2))
    else:
        raise ValueError("Invalid type of truncation.")


def make_df():
    sqt_n = 128
    n = sqt_n**2*3
    details = allcombinations({  # 'm' : 100,
        'm': logspace(2, n, 7, integer=True),
        "sparsity": "top",
    })
    first = pd.Dataframe({
        "sampling_type": "deterministic",
        "sampling_dist_type": "robust"
    })
    second = allcombinations({
        "sampling_type": "adjusted_Ber",
        "sampling_dist_type": ["optimized", "uniform"]
    }) 
    prelim = cross_merge_dfs(first, second)
    df = cross_merge_dfs(details, prelim)
    df = repeat(df, 4)
    global_seed = 1234
    np.random.seed(global_seed)
    df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
    add_standard_attributes(df)
    df.attrs.update(
        {'sqt_n': sqt_n, 'k': 1000, "max_iter": 1e4, "label": "sparse", "global_seed": global_seed, "save_img": False, "level": 3, "save_sampling": False,
         'sigma': 0.0,
         'img': "celeba",
         "preconditioned_optimization": True,
         "alg": "bpdn_projected",
         "memory": "3G",
         "num_cpus": "1",
         "num_gpus": "0",
         "time": "01:00:00",
         "rows_per_worker": 10})
    df = soft_astype_df(df, ExperimentTypes)
    return df


def wavelet_functions(signal_shape, level):
    axes = (len(signal_shape)-2, len(signal_shape)-1)
    sample_wavelet, slices = coeffs_to_array(
        wavedec2(np.zeros(signal_shape), "haar", level=level), axes=axes)
    theshape = sample_wavelet.shape

    def wavelet_transform(y) -> np.ndarray:  # 2d to
        return coeffs_to_array(wavedec2(y, "haar", level=level, axes=axes), axes=axes)[0].ravel()

    def inverse_wavelet(w: np.ndarray) -> np.ndarray:  # 1d to 2d
        return waverec2(array_to_coeffs(
            w.reshape(theshape), slices, output_format="wavedec2"), "haar")
    return (wavelet_transform, inverse_wavelet, np.prod(theshape))


def get_sparse_signal(sqt_n, img_name):
    # Generate the true signal
    if img_name == "phantom":
        true_signal = transform.resize(
            shepp_logan_phantom().astype(np.float32), (sqt_n, sqt_n))
    elif img_name == "camera":
        true_signal = transform.resize(
            camera().astype(np.float32), (sqt_n, sqt_n))
    elif img_name == "celeba":
        data_dir = os.getenv("DATA_DIR")
        assert data_dir is not None
        train_transforms = transforms.Compose([
            transforms.Resize((sqt_n, sqt_n)),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.CelebA(
            data_dir, target_type="identity", split="all", transform=train_transforms, download=False)
        true_signal, _ = dataset[np.random.randint(len(dataset))]
        true_signal = true_signal.numpy()
    else:
        raise NotImplementedError()
    return true_signal


def experiment(data: dict) -> dict[str, Any]:
    np.random.seed(data['seed'])
    start_time = time()
    sqt_n = int(data['sqt_n'])
    true_signal = get_sparse_signal(sqt_n, data['img'])
    # expected format
    assert len(true_signal.shape) == 2 or true_signal.shape[0] == 3
    wavelet_transform, inverse_wavelet, wav_len = wavelet_functions(
        true_signal.shape, data['level'])
    true_signal = truncate_to_sparsity(
        true_signal, data['k'], wavelet_transform, inverse_wavelet, data['sparsity'])
    true_signal = true_signal/np.linalg.norm(true_signal)

    # Override n because the wavelet transform may change it slightly.
    m = data['m']
    n = true_signal.size
    loc_co = Haar_local_coherence(true_signal.shape, data['level'])

    indices, d = sample_CS_matrix(
        loc_co, m, data['sampling_type'], data['sampling_dist_type'])

    m_tld = indices.size
    SF = SF2_func(indices, m, n)

    # Signal acquisition
    signal = SF(true_signal)
    noise = 1/np.sqrt(m) * data['sigma'] * np.random.randn(m_tld)
    b = signal + noise

    if data['preconditioned_optimization']:
        SDF = SDF2_func(indices, m, n, d)
        SDF_transpose = SDF2_transpose_map(indices, m, n, true_signal.shape, d)
        A = WaveletPartialFourier(indices, wav_len, lambda x: SDF(
            inverse_wavelet(x)), lambda x: wavelet_transform(SDF_transpose(x)))
        b = d[indices]*b
    else:
        SF_transpose = SF2_transpose_map(indices, m, n, true_signal.shape)
        A = WaveletPartialFourier(indices, wav_len, lambda x: SF(
            inverse_wavelet(x)), lambda x: wavelet_transform(SF_transpose(x)))

    # Recover the signal
    # lambda_ = data['lambda_coeff']*data['sigma']/np.sqrt(m) + 1e-8
    # x_haar = omp_with_linear_operator(A,b, data['k'])
    sigma_opt = max(data['sigma'], 1e-8)
    k = data['k']
    if data['alg'] == 'bpdn':
        x_haar, resid, grad, opt_info = spgl1.spg_bpdn(
            A, b, sigma_opt, iter_lim=data['max_iter'], verbosity=0)
        x_hat = inverse_wavelet(np.real(x_haar))
    elif data['alg'] == 'bpdn_projected':
        # First run BPDN to get initial solution
        x_haar, resid, grad, opt_info = spgl1.spg_bpdn(
            A, b, sigma_opt, iter_lim=data['max_iter'], verbosity=0)
        support = np.argsort(np.abs(x_haar))[-k:]
        A_support = np.zeros((m_tld, k), dtype=np.complex64)
        for i, idx in enumerate(support):
            basis_vector = np.zeros(n, dtype=np.complex64)
            basis_vector[idx] = 1
            A_support[:, i] = A.matvec(basis_vector)
        x_ls, _, _, _ = np.linalg.lstsq(A_support, b, rcond=None)
        x_haar = np.zeros_like(x_haar)
        x_haar[support] = x_ls
        x_hat = inverse_wavelet(np.real(x_haar))
        if np.linalg.norm(x_hat) > 2:  # limit magnitude of catastrophic error.
            x_hat = np.zeros_like(x_hat)
    else:
        raise ValueError("invalid alg argument")

    # Return relevant data.
    if data['sigma'] == 0:
        emp_snr = np.inf
    else:
        emp_snr = np.linalg.norm(signal)/np.linalg.norm(noise)
    # optim_info = {'resid': resid, 'grad': grad, 'info': info}
    re = np.linalg.norm(x_hat - true_signal)/np.linalg.norm(true_signal)
    run_time = time()-start_time
    return_dict = {'re': re, 'runtime_seconds': run_time, 'indices': indices, 'experiment_fn_label': "sparse_uneven",
                   "m_tld": m_tld, 'emp_snr': emp_snr, 'noise_mag': np.linalg.norm(noise), 'signal_mag': np.linalg.norm(signal)}
    if 'save_img' in data and data['save_img']:
        return_dict['recovered_signal'] = x_hat
    if 'save_sampling' in data and data['save_sampling']:
        return_dict['indices'] = indices
    return return_dict


if __name__ == "__main__":
    result = run_local_df_experiment(make_df(), experiment)
    result.to_pickle("./data/last_local.pickle")
