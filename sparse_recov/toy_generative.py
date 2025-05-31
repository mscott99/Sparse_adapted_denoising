import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import os
from time import time
from utils.utils_data import allcombinations, repeat
from torchvision import datasets, transforms
from sparse_recov.generative import ConvGenerator, Generator, VAEWrapper
from sparse_recov import convvae


def recover_signal(generator: Generator, encoder, target_signal: torch.Tensor, latent_dim: int, 
                   lr: float = 0.01, num_iterations: int = 1000, batch_size: int = 5, 
                   tol_stop: Optional[float] = 1e-7, beta1: float = 0.9, beta2: float = 0.999,
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                   log_interval: int = 100) -> Tuple[torch.Tensor, np.ndarray, float, bool, int]:
    """
    Recover the latent vector that generates a signal close to the target signal using gradient descent.
    
    Args:
        generator: Generator model (non-trainable) producing signals.
        target_signal: The target signal to match (tensor).
        latent_dim: Dimension of the latent space.
        lr: Learning rate for optimization.
        num_iterations: Number of optimization iterations.
        batch_size: Number of latent vectors to optimize in parallel (best of batch_size is chosen).
        tol_stop: Tolerance for early stopping based on loss.
        beta1, beta2: Adam optimizer parameters.
        device: Device to perform computation on.
        log_interval: Interval at which to log the loss during optimization.
    
    Returns:
        recovered_signal: Recovered signal as a torch tensor.
        losses: Array of loss values over iterations (logged at log_interval).
        smallest_loss: Final smallest loss value achieved.
        tol_stopped: Boolean indicating if optimization stopped due to tolerance.
        last_iter: Number of iterations performed.
    """
    signal_size = torch.norm(target_signal).detach().cpu().numpy()
    generator.to(device)
    for param in generator.parameters():
        param.requires_grad = False

    target_signal = target_signal.to(device).unsqueeze(0)
    z = encoder(target_signal)
    z.requires_grad= True
    # z = torch.randn(batch_size, latent_dim, requires_grad=True, device=device)
    optimizer = optim.Adam([z], lr=lr, betas=(beta1, beta2))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-7)

    # optimizer = optim.LBFGS([z], lr=lr)
    # def closure():
    #         optimizer.zero_grad()
    #         generated_signal = generator(z)
    #         current_loss = torch.sqrt(torch.sum((generated_signal - target_signal)**2))
    #         current_loss.backward() # Sum if it's a batch
    #         return current_loss # LBFGS needs a scalar loss 

    losses = []
    tol_stopped = False
    last_iter = 0

    for i in range(num_iterations):
        optimizer.zero_grad()
        # generated_signal = generator(z)
        # loss_array = torch.sum((generated_signal - target_signal) ** 2, dim=tuple(range(1, target_signal.ndim)), keepdim=False)
        # if tol_stop is not None and loss_array.min() < tol_stop:
        #     tol_stopped = True
        #     last_iter = i
        #     break
        # loss = loss_array.sum()
        generated_signal = generator(z)
        loss_array = torch.sum(torch.abs(generated_signal - target_signal), dim=tuple(range(1, target_signal.ndim)), keepdim=False)
        if i % log_interval == 0:
            losses.append(loss_array.mean().detach().cpu().numpy()/signal_size)
        # loss.backward()
        optimizer.step()
        # scheduler.step(loss_array.min())
        # optimizer.step(closure)
        
    if last_iter == 0:
        last_iter = num_iterations

    with torch.no_grad():
        generated_signal = generator(z)
        loss = torch.sum((generated_signal - target_signal) ** 2, dim=tuple(range(1, target_signal.ndim)), keepdim=False)
        best = torch.argmin(loss)
        smallest_loss = torch.min(loss).item()
        recovered_signal = generator(z[best, :].unsqueeze(0)).squeeze(0)

    return recovered_signal, np.array(losses), smallest_loss, tol_stopped, last_iter

def make_df():
    """
    Create a DataFrame with experiment configurations for signal recovery.
    
    Returns:
        pd.DataFrame: DataFrame containing experiment parameters.
    """
    df = allcombinations({
        'lr': [0.1, 1.0, 10.0, 0.0005],  # Learning rate for optimization
        'beta1': [0.9],  # Adam optimizer parameter
        'beta2': [0.999],  # Adam optimizer parameter
        'max_iter': [10000],  # Maximum number of iterations
        'opt_tol_stop': [1e-8],  # Tolerance for early stopping
        'num_repeats_in_optim': [1],  # Batch size for optimization (number of parallel attempts)
        'model_type': ['conv_celeba'],  # Model type for the generator
        'dataset': ['CELEBA'],  # Dataset used for signal generation
    })
    df = repeat(df, 3)
    global_seed = 1234
    np.random.seed(global_seed)
    df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
    df.attrs.update({
        'label': 'toy_generative_inversion',
        'global_seed': global_seed,
        'save_img': True,
        'save_opt_data': True
    })
    return df

def experiment(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run an experiment to generate a random signal using the CelebA model and recover it by inverting the generator.
    
    Args:
        data: Dictionary containing experiment parameters.
    
    Returns:
        Dict[str, Any]: Dictionary with experiment results.
    """
    np.random.seed(data['seed'])
    start_time = time()
    return_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the CelebA generative model
    full_model = convvae.get_model()
    generator = VAEWrapper(full_model).to(device)
    encoder = lambda x: full_model.forward_encoder(x)[0]

    # Generate a random target signal
    with torch.no_grad():
        target_signal = generator(encoder(generator(torch.randn(1, generator.k).to(device)))).squeeze(0)

    # Recover the signal by inverting the generator
    recovered_signal, opt_data, smallest_loss, tol_stopped, last_iter = recover_signal(
        generator, encoder, target_signal, generator.k, data['lr'], data['max_iter'], 
        data['num_repeats_in_optim'], tol_stop=data['opt_tol_stop'], 
        beta1=data['beta1'], beta2=data['beta2'], device=device
    )

    target_signal_np = target_signal.cpu().numpy()
    recovered_signal_np = recovered_signal.cpu().numpy()

    # Compute relative error
    re = np.linalg.norm(recovered_signal_np - target_signal_np) / np.linalg.norm(target_signal_np)
    run_time = time() - start_time

    return_dict = {
        're': re,
        'runtime_seconds': run_time,
        'experiment_fn_label': 'toy_generative_inversion',
        'smallest_loss': smallest_loss,
        'tol_stopped': tol_stopped,
        'last_iter': last_iter
    }

    if 'save_img' in data and data['save_img']:
        return_dict['recovered_signal'] = recovered_signal_np
        return_dict['target_signal'] = target_signal_np
    if 'save_opt_data' in data and data['save_opt_data']:
        return_dict['opt_data'] = opt_data

    return return_dict
