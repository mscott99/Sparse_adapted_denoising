import torch
from utils_data import cross_merge_dfs, logspace, allcombinations, add_standard_attributes, repeat, soft_astype_df
from time import time
import pandas as pd
from typing import TypedDict, Literal, Any, Tuple, List, Optional, Union
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator
from typing import Tuple, Callable
import pickle
import os
from .train_model import Generator, ConvVAE

from sparse_recov.sampling_recov_functions import sample_CS_matrix


def generate_random_signal(generator: Generator, latent_dim: int, device: str = 'cpu') -> torch.Tensor:
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        return generator(z)


def get_fft_loc_coherences(generator: Generator, num_samples: int = 50,
                           device: str = 'cpu') -> np.ndarray:
    with torch.no_grad():
        codes = torch.randn(num_samples, generator.k).to(device)
        outputs = generator(codes)  # Shape: (num_samples, 1, 28, 28)
        # Remove channel dimension (new Generator outputs shape [batch, 1, H, W])
        outputs = outputs.squeeze(1)  # Shape: (num_samples, 28, 28)

        outputs_i = outputs.unsqueeze(1)  # (num_samples, 1, 28, 28)
        outputs_j = outputs.unsqueeze(0)  # (1, num_samples, 28, 28)
        # (num_samples, num_samples, 28, 28)
        differences = outputs_i - outputs_j
        norms = torch.norm(differences, dim=(2, 3), keepdim=True)
        normalized_diffs = differences / \
            (norms + 1e-8)  # Avoid division by zero

        # Step 5: Compute FFT of the normalized differences
        # Convert to complex for FFT
        coefs = torch.fft.fft2(normalized_diffs, norm="ortho")
        loc_co = torch.max(torch.abs(coefs), dim=0)[
            0]  # (num_samples^2, 28, 28)
        loc_co = torch.max(loc_co, dim=0)[0]
    return loc_co.cpu().numpy()  # Return as numpy array


class CSMeasurementMatrix(nn.Module):
    """
    A PyTorch module for a differentiable compressed sensing measurement matrix.
    Applies a subsampled 2D Fourier transform based on given indices.

    Args:
        indices: Tensor of sampled indices (long tensor)
        m: Number of measurements (desired)
        n: Signal dimension (e.g., 784 for 28x28)
        shape: 2D shape of the signal (e.g., (28, 28))
        preconditioned: Whether to apply preconditioning (requires d)
        d: Optional preconditioning weights (default None)
    """

    def __init__(self, indices, m: int, signal_shape: Tuple, ):
        super(CSMeasurementMatrix, self).__init__()
        self.m = m  # Desired number of measurements
        self.signal_shape = signal_shape  # Signal dimension
        indices = torch.tensor(indices)
        self.register_buffer('indices', indices.long())
        self.m_tld = indices.size(0)  # Actual number of measurements
        n = np.prod(signal_shape)
        # Preconditioning weights (if applicable)

        # Scaling factor for normalization (consistent with SF2_func)
        self.scale = torch.sqrt(torch.tensor(n / m, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the measurement matrix to the input signal.

        Args:
            x: Input tensor of shape (batch, channel, height, width), e.g., (batch, 1, 28, 28)

        Returns:
            measurements: Tensor of shape (batch, m_tld) with complex values
        """
        batch_size = x.size(0)

        # Handle case with channel dimension (new Generator output)
        if len(x.shape) == 4:  # (batch, channel, height, width)
            assert x.shape[2:
                           ] == self.signal_shape, f"Input shape {x.shape[2:]} must match {self.signal_shape}"
            # Use only the first channel for FFT
            x = x.squeeze(1)  # (batch, height, width)
        else:  # Original case (batch, height, width)
            assert x.shape[1:
                           ] == self.signal_shape, f"Input shape {x.shape[1:]} must match {self.signal_shape}"

        # (batch, height, width), complex
        x_fft = torch.fft.fft2(x, norm="ortho")
        # (batch, n), e.g., (batch, 784)
        x_fft_flat = x_fft.view(batch_size, -1)
        # Subsample using indices
        measurements = self.scale * \
            x_fft_flat[:, self.indices]  # (batch, m_tld)
        # Apply preconditioning if enabled
        return measurements


class preconditionedCSMeasurementMatrix(nn.Module):
    def __init__(self, SF: CSMeasurementMatrix, dtld: torch.Tensor):
        super(preconditionedCSMeasurementMatrix, self).__init__()
        self.SF = SF
        self.dtld = dtld

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dtld * self.SF.forward(x)


def recover_signal(generator: Generator, A: Union[CSMeasurementMatrix, preconditionedCSMeasurementMatrix], b: torch.Tensor,
                   k: int, lr: float = 0.01, num_iterations: int = 40000, batch_size=5, tol_stop: Optional[float] = 1e-7, beta1=0.9, beta2=0.999, true_signal=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> Tuple[torch.Tensor, NDArray, NDArray, float]:
    """
    Recover the signal by optimizing the latent vector z using the generative prior, the best of batch_size attempts.

    Args:
        generator: Generator model (non-trainable) producing 2D signals
        measurements: Observed measurements (tensor, shape (m_tld,) or (1, m_tld))
        A: CSMeasurementMatrix instance (non-trainable) for differentiable measurements
        latent_dim: Dimension of the latent space (generator.k)
        lr: Learning rate for optimization (default 0.01)
        num_iterations: Number of optimization iterations (default 1000)

    Returns:
        recovered_signal: Recovered signal as a numpy array (e.g., 28x28)
        z_opt: Optimized latent vector as a torch tensor
    """
    # Move models to device
    generator.to(device)
    A.to(device)

    for param in generator.parameters():
        param.requires_grad = False

    # Initialize latent vector z from N(0,1) as a trainable parameter
    z = torch.randn(batch_size, k, requires_grad=True, device=device)
    optimizer = optim.Adam([z], lr=lr, betas=(beta1, beta2))
    losses = []
    test_losses = []
    true_signal = true_signal
    true_sig_norm = torch.norm(true_signal)
    # Optimization loop
    for _ in range(num_iterations):
        optimizer.zero_grad()
        x_tensor = generator(z)  # Shape: (1, height, width), e.g., (1, 28, 28)
        offset = A(x_tensor) - b
        loss = torch.sum(offset.real**2 + offset.imag**2)
        losses.append(loss.detach().cpu().numpy())
        test_losses.append(
            (torch.norm(x_tensor - true_signal)/true_sig_norm).item())
        if tol_stop != None and loss < tol_stop:
            break
        # Backpropagate and update z
        loss.backward()
        optimizer.step()

    # Final signal recovery
    with torch.no_grad():
        x_tensor = generator(z)  # Shape: (1, height, width), e.g., (1, 28, 28)
        offset = A(x_tensor) - b
        loss = torch.sum(offset.real**2 + offset.imag**2, dim=1, keepdim=False)
        best = torch.argmax(loss)
        smallest_loss = torch.max(loss).item()
        # Get recovered signal and remove channel dimension if present
        # Shape: (1, 1, height, width)
        recovered = generator(z[best, :].unsqueeze(0))
        if recovered.dim() == 4:
            recovered_signal = recovered.squeeze(
                0).squeeze(0)  # Shape: (height, width)
        else:
            recovered_signal = recovered.squeeze(0)  # Shape: (height, width)

    return recovered_signal, np.array(losses), np.array(test_losses), smallest_loss


ExperimentTypes = {
    'sqt_n': int, 'k': int, 'm': int, 'm_tld': int, 'sigma': float,
    'lr': float, 'max_iter': int, 'num_repeats_in_optim': int,
    'sampling_type': pd.CategoricalDtype(['wr', 'Ber', 'wor']),
    'sampling_dist_type': pd.CategoricalDtype(['uniform', 'optimized', 'robust', 'half_robust']),
    'emp_snr': float, 'preconditioned_optimization': bool,
    "opt_tol_stop": float,
    "beta1": float,
    "beta2": float,
    'model_type': pd.CategoricalDtype(["perturbed", "trained"]),
}


def make_df():
    sqt_n = 28  # Fixed image size, matching sparse experiment
    df = allcombinations({
        # Number of measurements
        'm': logspace(2, sqt_n**2/2, 7, integer=True),
        'sigma': [0.0, 0.1, 0.3, 1.0],  # Noise levels
        # Weighted random sampling (fixed as in generative.py)
        # robust sampling schemes
        # 'sampling_type': ['wr'],
        # 'sampling_dist_type': ['optimized', 'robust', 'half_robust'],
        # 'preconditioned_optimization': [True, False],
        'sampling_type': ['wr', 'wor', 'Ber'],
        'sampling_dist_type': 'optimized',
        'preconditioned_optimization': [False],
        'lr': [0.03],  # Learning rate for optimization
        'beta1': [0.9],
        'beta2': [0.999],
        'max_iter': [1000],  # 15000 more than enough for MNIST recovery.
        'opt_tol_stop': [1e-7],
        # Number of optimization repeats (batch size in recover_signal)
        'num_repeats_in_optim': [1],
    })
    
    model_df = pd.DataFrame({'model_type': ["perturbed", "trained"], 'k': [16, 8]})
    df = cross_merge_dfs(df, model_df)

    df = repeat(df, 10)

    df['sampling_type'] = df['sampling_type'].astype('category')
    global_seed = 1234
    np.random.seed(global_seed)
    df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
    add_standard_attributes(df)
    df.attrs.update({
        'sqt_n': sqt_n,
        'label': 'generative_vanilla',
        'global_seed': global_seed,
        'save_img': False,
        'save_sampling': False,
        'save_opt_data': True
    })
    add_standard_attributes(df)
    df = soft_astype_df(df, ExperimentTypes)
    return df

# def make_df():
#     sqt_n = 32  # Fixed image size, matching sparse experiment
#     df = allcombinations({
#         'm': logspace(3e2, sqt_n**2 * 4, 8, integer=True),  # Number of measurements
#         'k': [60],  # Latent dimension (analogous to sparsity level)
#         'sigma': [0.0, 0.01, 0.3, 1.0],  # Noise levels
#         'sampling_type': 'wr',  # Weighted random sampling (fixed as in generative.py)
#         'sampling_dist_type': ['uniform', 'optimized', 'robust'],  # Sampling distribution types
#         'preconditioned_optimization': [True, False],  # Preconditioning option
#         'lr': [0.001],  # Learning rate for optimization
#         'max_iter': [5000],  # Number of iterations for recovery
#         'num_repeats_in_optim': [2]  # Number of optimization repeats (batch size in recover_signal)
#     })
#
#     # Repeat experiments for statistical robustness
#     df = repeat(df, 5)
#
#     # Set categorical type for sampling_type
#     df['sampling_type'] = df['sampling_type'].astype('category')
#
#     # Assign random seeds
#     global_seed = 1234
#     np.random.seed(global_seed)
#     df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
#
#     # Add standard attributes
#     add_standard_attributes(df)
#
#     # Update DataFrame attributes with experiment-specific settings
#     df.attrs.update({
#         'sqt_n': sqt_n,
#         'max_iter': 1e4,  # Maximum iterations (though num_iter is used in recover_signal)
#         'label': 'generative_recovery',
#         'global_seed': global_seed,
#         'save_img': False,
#         'save_sampling': False
#     })
#
#     # Define experiment types for type consistency (adapted from sparse case)
#     ExperimentTypes = {
#         'sqt_n': int, 'k': int, 'm': int, 'm_tld': int, 'sigma': float,
#         'lr': float, 'max_iter': int, 'num_repeats_in_optim': int,
#         'sampling_type': pd.CategoricalDtype(['wr', 'Ber']),
#         'sampling_dist_type': pd.CategoricalDtype(['uniform', 'optimized', 'robust']),
#         'emp_snr': float, 'preconditioned_optimization': bool
#     }
#
#     # Apply type casting
#     df = soft_astype_df(df, ExperimentTypes)
#
#     return df


def load_saved_model(k: int, file_name: str):
    proj_path = os.getenv("PROJECT")
    if proj_path == None:
        raise Exception("PROJECT env var was not set.")
    model = ConvVAE(k=k)
    model.load_state_dict(torch.load(os.path.join(
        proj_path, "models", file_name), map_location=torch.device('cpu')))
    return model.decoder


def experiment(data: dict) -> dict[str, Any]:
    np.random.seed(data['seed'])
    start_time = time()
    sqt_n = int(data['sqt_n'])
    n = sqt_n**2
    m = data['m']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data['model_type'] == "random":
        generator = Generator(data["k"], (sqt_n, sqt_n), [256, 512, 1024])
    elif data['model_type'] == "trained":
        assert data['k'] == 8
        generator = load_saved_model(k=8, file_name="model_weights.pth")
    elif data['model_type'] == "perturbed":
        assert data['k'] == 16
        generator = load_saved_model(
            k=16, file_name="perturbed_model_weights_v4.pth")
    else:
        raise ValueError("model_type can only be 'random' or 'trained'")
    # Override n because the wavelet transform may change it slightly
    loc_co = get_fft_loc_coherences(generator)
    indices, d = sample_CS_matrix(
        loc_co, m, data['sampling_type'], data['sampling_dist_type'])
    if "prec_exponent" in data:
        d = (d**(-2))**data["prec_exponent"]
    d = torch.tensor(d).to(device)
    m_tld = indices.size

    true_signal = generate_random_signal(generator, data['k']).to(device)

    # Signal acquisition
    SF = CSMeasurementMatrix(indices, m, (sqt_n, sqt_n)).to(device)
    signal = SF(true_signal)
    noise = torch.tensor(
        1/np.sqrt(m) * data['sigma'] * np.random.randn(m_tld)).to(device)
    b = signal + noise

    SDF = SF
    btld = b
    if 'preconditioned_optimization' in data.keys() and data['preconditioned_optimization']:
        if data['sampling_type'] == 'Ber':
            raise NotImplementedError(
                "Bernoulli sampling with preconditioned opt not implemented")
        dtld = d[indices]
        SDF = preconditionedCSMeasurementMatrix(SF, dtld)
        btld = dtld*b

    x_hat, opt_data, test_losses, smallest_loss = recover_signal(
        generator, SDF, btld, data['k'], data['lr'], data['max_iter'], data['num_repeats_in_optim'], tol_stop=data['opt_tol_stop'], beta1=data["beta1"], beta2=data["beta2"], true_signal=true_signal, device=device)

    true_signal = true_signal.cpu().numpy()
    x_hat = x_hat.cpu().numpy()
    # Return relevant data.
    signal_mag = np.linalg.norm(signal.cpu().numpy())
    noise_mag = np.linalg.norm(noise.cpu().numpy())
    if noise_mag == 0:
        emp_snr = np.inf
    else:
        emp_snr = signal_mag / noise_mag
    # optim_info = {'resid': resid, 'grad': grad, 'info': info}
    re = np.linalg.norm(x_hat - true_signal)/np.linalg.norm(true_signal)
    run_time = time()-start_time
    return_dict = {'re': re, 'runtime_seconds': run_time, 'indices': indices, 'experiment_fn_label': "uneven_on_exactly_sparse", "m_tld": m_tld,
                   'emp_snr': emp_snr, 'noise_mag': noise_mag, 'signal_mag': signal_mag, "smallest_loss": smallest_loss}
    if 'save_img' in data and data['save_img']:
        return_dict['recovered_signal'] = x_hat
    if 'save_sampling' in data and data['save_sampling']:
        return_dict['indices'] = indices
    if 'save_opt_data' in data and data['save_opt_data']:
        return_dict['opt_data'] = opt_data
        return_dict['test_losses'] = test_losses
    return return_dict
