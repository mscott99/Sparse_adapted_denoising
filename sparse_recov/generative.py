import torch
import torchvision
from torchvision import datasets, transforms, utils
from utils.utils_data import cross_merge_dfs, logspace, allcombinations, add_standard_attributes, repeat, soft_astype_df
from time import time
import pandas as pd
from typing import Any, Tuple, Optional, Callable, Optional
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import os
from .train_model import ConvGenerator, Generator, ConvVAE, VAEWrapper
from . import convvae
from .optimization import exterior_redo_recover_signal

from sparse_recov.sampling_recov_functions import sample_CS_matrix


def generate_random_signal(generator: Generator, latent_dim: int, device=torch.device("cuda") if torch.cuda.is_available() else "cpu") -> torch.Tensor:
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        return generator(z)


def get_fft_loc_coherences(generator: Generator, num_samples: int = 50,
                           device=torch.device(
                               "cuda" if torch.cuda.is_available() else "cpu")
                           ) -> np.ndarray:
    with torch.no_grad():
        codes = torch.randn(num_samples, generator.k).to(device)
        outputs = generator(codes)  # Shape: (num_samples, 1, 28, 28)
        # Remove channel dimension (new Generator outputs shape [batch, 1, H, W])
        # outputs = outputs.squeeze(1)  # Shape: (num_samples, 28, 28)

        outputs_i = outputs.unsqueeze(1)  # (num_samples, 1, 28, 28)
        outputs_j = outputs.unsqueeze(0)  # (1, num_samples, 28, 28)
        # (num_samples, num_samples, 28, 28)
        differences = outputs_i - outputs_j
        norms = torch.norm(differences, p=2, dim=(2, 3, 4), keepdim=True)
        normalized_diffs = differences / \
            (norms + 1e-8)  # Avoid division by zero

        # Step 5: Compute FFT of the normalized differences
        # Convert to complex for FFT
        coefs = torch.fft.fft2(normalized_diffs, norm="ortho")
        loc_co = torch.max(torch.abs(coefs), dim=0)[
            0]  # (num_samples^2, 28, 28)
        loc_co = torch.max(loc_co, dim=0)[0]
    return loc_co.cpu().numpy()  # Return as numpy array


class SubsampledFourierMatrix(nn.Module):
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
        super(SubsampledFourierMatrix, self).__init__()
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
            assert x.shape[1:
                           ] == self.signal_shape, f"Input shape {x.shape[1:]} must match {self.signal_shape}"
            # Use only the first channel for FFT
            x = x.squeeze(1)  # (batch, height, width)
        else:  # Original case (batch, height, width)
            assert x.shape == self.signal_shape, f"Input shape {x.shape} must match {self.signal_shape}"

        # (batch, height, width), complex
        x_fft = torch.fft.fft2(x, norm="ortho")
        # (batch, n), e.g., (batch, 784)
        x_fft_flat = x_fft.view(batch_size, -1)
        # Subsample using indices
        measurements = self.scale * \
            x_fft_flat[:, self.indices]  # (batch, m_tld)
        # Apply preconditioning if enabled
        return measurements


class preconditionedSubsampledFourierMatrix(nn.Module):
    def __init__(self, SF: SubsampledFourierMatrix, dtld: torch.Tensor):
        super(preconditionedSubsampledFourierMatrix, self).__init__()
        self.SF = SF
        self.dtld = dtld

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dtld * self.SF.forward(x)


ExperimentTypes = {
    'sqt_n': int, 'k': int, 'm': int, 'm_tld': int, 'sigma': float,
    'lr': float, 'max_iter': int, 'num_repeats_in_optim': int,
    'sampling_type': pd.CategoricalDtype(['wr', 'Ber', 'wor', 'adjusted_Ber', 'deterministic']),
    'preconditioner': pd.CategoricalDtype(['wr', 'Ber', 'wor', 'adjusted_Ber', 'deterministic', 'empirical', 'cond_poisson']),
    'sampling_dist_type': pd.CategoricalDtype(['uniform', 'optimized', 'robust', 'half_robust', 'deterministic']),
    'emp_snr': float, 'preconditioned_optimization': bool,
    "opt_tol_stop": float,
    "beta1": float,
    "beta2": float,
    # pd.CategoricalDtype(["perturbed", "trained", "heavy_tail", "co_heavy_tail", "conv_celeba"])
    'model_type': 'str',
    'true_signal_dist': pd.CategoricalDtype(["train", "test", "random"]),
    'dataset': pd.CategoricalDtype(["CELEBA", "MNIST"]),
    'project_to_prior': bool,
}


def make_df():
    df = allcombinations({
        # Number of measurements
        # 'm': 28**2,
        # 'm': [2,8,64],
        'm': logspace(int(2), 128*112*3, 7, integer=True),
        # 'm': 3*128*112,  # logspace(n/2, n, 3, integer=True),
        # 'm':10,
        # 'sigma': [0.0, 0.1, 0.5, 1.0],  # Noise levels
        'sigma': 0.0,
        # Weighted random sampling (fixed as in generative.py)
        # robust sampling schemes
        # 'sampling_type': ['wr'],
        # 'preconditioned_optimization': [True, False],
        # 'sampling_type': ['wr', 'wor', 'Ber', 'adjusted_Ber'],
        'true_signal_dist': 'random',
        # 'project_to_prior': False,
        'preconditioned_optimization': [True],
        'preconditioner': ["empirical", "Ber", "cond_poisson"],
        'lr': [1.0],  # Learning rate for optimization
        'beta1': [0.9],
        'beta2': [0.999],
        'max_iter': [4],  # Maximum number of iterations
        'opt_tol_stop': [0.0001],
        # Batch size for optimization (number of parallel attempts)
        'num_repeats_in_optim': [5],
        # Number of optimization repeats (batch size in recover_signal)
        # 'model_type': ['to_the_ground_no_co', 'higher_co_02'],  # Model type for the generator
        'model_type':'to_the_ground_no_co',
        'sampling_type': ['wor'],
        'sampling_dist_type': 'optimized',
    })
    df = repeat(df, 20)
    # df['sampling_type'] = df['sampling_type'].astype('category')
    global_seed = 1234
    np.random.seed(global_seed)
    df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
    df.attrs.update({
        # 'sqt_n': 28,
        # 'n': 256**2*3,
        'k': 200,
        'n': 128*112*3,
        'label': 'generative_vanilla',
        'global_seed': global_seed,
        'save_img': True,
        'optimizer': 'LBFGS',
        'save_sampling': False,
        'save_opt_data': False,
        'dataset': 'CELEBA',
        "memory": "3G",
        "num_cpus": "2",
        "num_gpus": "1",
        "time": "01:00:00",
        "rows_per_worker": 500
    })
    add_standard_attributes(df)
    df = soft_astype_df(df, ExperimentTypes)
    return df


def load_saved_model(k: int, file_name: str):
    # proj_path = os.getenv("PROJECT")
    model_dir = os.getenv("MODEL_DIR")
    if model_dir == None:
        raise Exception("MODEL_DIR env var was not set.")
    model = ConvVAE(k=k)
    model.load_state_dict(torch.load(os.path.join(
        model_dir, file_name), map_location=torch.device('cpu')))
    return model.decoder


def experiment(data: dict) -> dict[str, Any]:
    np.random.seed(data['seed'])
    start_time = time()
    return_dict = {}
    m = data['m']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape=None
    if data['model_type'] == "random":
        sqt_n = int(data['sqt_n'])
        input_shape = (1, sqt_n, sqt_n)
        generator = ConvGenerator(data["k"], (sqt_n, sqt_n), [256, 512, 1024])
    elif data['model_type'] == "trained":
        assert data['k'] == 8
        generator = load_saved_model(k=8, file_name="model_weights.pth")
        input_shape=(1,28,28)
    elif data['model_type'] == "perturbed":
        assert data['k'] == 16
        generator = load_saved_model(
            k=16, file_name="perturbed_model_weights_v4.pth")
        input_shape=(1,28,28)
    elif data['model_type'] == "heavy_tail":
        assert data['k'] == 16
        generator = load_saved_model(
            k=16, file_name="fat_tail.pth")
        input_shape=(1,28,28)
    elif data['model_type'] == "co_heavy_tail":
        assert data['k'] == 16
        generator = load_saved_model(
            k=16, file_name="heavy_tail_co_v0.pth")
        input_shape=(1,28,28)
    elif data['model_type'] == "MNIST_vae_16":
        generator = load_saved_model(
            k=16, file_name="MNIST_vae_16.pth")
        input_shape=(1,28,28)
    elif data['model_type'] == "conv_celeba":
        generator = VAEWrapper(convvae.get_model())
        input_shape = generator.output_shape
    else:  # Here model_type is basename of file pth.
        full_model = convvae.get_model(data['model_type']).eval()
        generator = VAEWrapper(full_model).to(device)
        input_shape = generator.output_shape
    return_dict['k'] = data['k'] = generator.k
    generator.to(device)
    # Override n because the wavelet transform may change it slightly
    loc_co = get_fft_loc_coherences(generator, device=device)
    n=loc_co.size
    indices, d = sample_CS_matrix(
        loc_co, m, data['sampling_type'], data['sampling_dist_type'], data['preconditioner'])
    if "prec_exponent" in data:
        d = (d**(-2))**data["prec_exponent"]
    d = torch.tensor(d).to(device)
    m_tld = indices.size

    if "true_signal_dist" not in data or data["true_signal_dist"] == "random":
        true_signal = generate_random_signal(generator, data['k']).to(device).squeeze(0)
    elif data["true_signal_dist"] in ["test", "train"]:
        # raise NotImplementedError("Only random is implemented")
        assert "dataset" in data
        if data["dataset"] == "CELEBA":
            train_transforms = transforms.Compose([
                transforms.Resize((128, 112)),
                transforms.ToTensor(),
            ])
            data_dir = os.getenv("DATA_DIR")
            assert data_dir is not None
            train_data = torchvision.datasets.CelebA(data_dir, target_type = "identity", split=data["true_signal_dist"], transform = train_transforms, download = False)
            random_idx = np.random.randint(0, int(len(train_data)))
            true_signal, _ = train_data[random_idx]
            true_signal = true_signal.to(device)
            # full_data = datasets.ImageFolder(
            #     "./datasets/celeba", transform=train_transforms)
            # # Sample one random image from the dataset
            #     random_idx = np.random.randint(0, int(len(full_data)*0.8))
            # else:
            #     random_idx = np.random.randint(
            #         int(len(full_data)*0.8), len(full_data))
            if "project_to_prior" in data and data['project_to_prior']:
                true_signal, _, _, _, _, _ = exterior_redo_recover_signal(generator, true_signal, data['k'], lr = data['lr'], num_tries=data['num_repeats_in_optim'], max_iter=data['max_iter'], tol_stop=data['opt_tol_stop'], device=device)
            # trainloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=True)
        else:
            raise NotImplementedError("Implement dataset: ", data["dataset"])
    else:
        raise NotImplementedError()

    # Signal acquisition
    SF = SubsampledFourierMatrix(indices, m, input_shape).to(device)
    signal = SF(true_signal.unsqueeze(0)).squeeze(0)
    noise = torch.tensor(
        1/np.sqrt(m) * data['sigma'] * np.random.randn(m_tld)).to(device)
    b = signal + noise

    SDF = SF
    btld = b
    if 'preconditioned_optimization' in data.keys() and data['preconditioned_optimization']:
        # if data['sampling_type'] == 'Ber':
        #     raise NotImplementedError(
        #         "Bernoulli sampling with preconditioned opt not implemented")
        dtld = d[indices]
        SDF = preconditionedSubsampledFourierMatrix(SF, dtld)
        btld = dtld*b
    SDF.to(device)
    btld.to(device)
    x_hat, opt_data, test_losses, smallest_loss, tol_stopped, last_opt_iter = exterior_redo_recover_signal(
        generator, btld, latent_dim = data['k'], A = SDF, max_iter = data['max_iter'], lr = data['lr'], num_tries=data['num_repeats_in_optim'], tol_stop=data['opt_tol_stop'], beta1=data["beta1"], beta2=data["beta2"], true_signal=true_signal, device=device)

    #return_dict['reconstructed_img'] = full_model.forward(true_signal.unsqueeze(0))[2].squeeze(0).cpu().numpy()
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
    return_dict.update({'re': re, 'runtime_seconds': run_time, 'indices': indices, 'experiment_fn_label': "generative", "m_tld": m_tld,
                   'emp_snr': emp_snr, 'noise_mag': noise_mag, 'signal_mag': signal_mag, "smallest_loss": smallest_loss, "tol_stopped": tol_stopped,
                   "last_tol_iter": last_opt_iter, 'n': n})
    if 'save_img' in data and data['save_img']:
        return_dict['recovered_signal'] = x_hat
        return_dict['true_signal'] = true_signal
    if 'save_sampling' in data and data['save_sampling']:
        return_dict['indices'] = indices
    if 'save_opt_data' in data and data['save_opt_data']:
        return_dict['opt_data'] = opt_data
        return_dict['test_losses'] = test_losses

    return return_dict
