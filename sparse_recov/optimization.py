import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, Callable
import os
from time import time
from utils.utils_data import allcombinations, repeat
from torchvision import datasets, transforms
from sparse_recov.train_model import Generator, VAEWrapper
from sparse_recov import convvae

def exterior_redo_recover_signal(*args, num_tries=3, tol_stop = 1e-4, batch_size=1, **kwargs):
    best_obj = recover_signal(*args, **kwargs)
    for i in range(1,num_tries):
        last_obj = recover_signal(*args, batch_size=batch_size, tol_stop = tol_stop, **kwargs)
        if last_obj[3] < best_obj[3]:
            best_obj = last_obj
            if best_obj[3] < tol_stop:
                break
    return best_obj

def recover_signal(generator: Generator,  b: torch.Tensor, latent_dim: int, A:Callable = lambda x: x,
                   lr: float = 0.01, max_iter: int = 1000, batch_size: int = 1, 
                   tol_stop: Optional[float] = 1e-7, true_signal=None, beta1: float = 0.9, beta2: float = 0.999,
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), encoder=None, 
                   log_interval: int = 3, starting_code:Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, float, bool, int]:
    """
    Recover the latent vector that generates a signal close to the target signal using gradient descent.
    
    Args:
        generator: Generator model (non-trainable) producing signals.
        target_signal: The target signal to match (tensor).
        latent_dim: Dimension of the latent space.
        lr: Learning rate for optimization.
        max_iter: Number of optimization iterations.
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
    generator.requires_grad_(False)
    generator.to(device)
    generator.eval()

    b = b.to(device).unsqueeze(0)
    b.requires_grad=False
    measurement_size = torch.norm(b).item()

    if starting_code != None:
        z = starting_code
    elif encoder != None:
        z = encoder(b)
        z.requires_grad = True
    else:
        z = torch.randn(batch_size, latent_dim, requires_grad=True, device=device)

    # optimizer = optim.Adam([z], lr=lr, betas=(beta1, beta2))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-7)

    optimizer = optim.LBFGS([z], lr=lr)
    def closure():
            optimizer.zero_grad()
            offset = A(generator(z)) - b
            if offset.is_complex():
                loss = torch.sum(offset.real**2 + offset.imag**2)
            else:
                loss = torch.sum(offset**2)
            loss.backward() # Sum if it's a batch
            return loss # LBFGS needs a scalar loss 

    losses = []
    test_losses = []
    if true_signal is not None:
        true_signal = torch.unsqueeze(true_signal, 0) 
    tol_stopped = False
    last_iter = 0
    for i in range(max_iter):
        # generated_signal = generator(z)
        # loss_array = torch.sum((generated_signal - target_signal) ** 2, dim=tuple(range(1, target_signal.ndim)), keepdim=False)
        # loss = loss_array.sum()
        # optimizer.zero_grad()
        if i % log_interval == 0:
            generated_signal = generator(z)
            loss_array = torch.sum(torch.abs(A(generated_signal) - b)**2, dim=tuple(range(1, b.ndim)), keepdim=False)
            min_normed_loss = torch.sqrt(loss_array.min()).item()/measurement_size
            if tol_stop is not None and min_normed_loss < tol_stop:
                tol_stopped = True
                last_iter = i
                break
            losses.append(min_normed_loss)
            if true_signal is not None:
                true_loss = torch.min(torch.sum(torch.abs(generated_signal - true_signal)**2, dim=tuple(range(1, b.ndim)), keepdim=False))
                test_losses.append(true_loss.item())
        # loss = loss_array.sum()
        # loss.backward()
        # optimizer.step()
        # scheduler.step(loss_array.min())
        optimizer.step(closure)
        
    if last_iter == 0:
        last_iter = max_iter

    with torch.no_grad():
        loss_array = torch.sum(torch.abs(A(generator(z)) - b) ** 2, dim=tuple(range(1, b.ndim)), keepdim=False)
        best = torch.argmin(loss_array)
        smallest_loss =  torch.sqrt(loss_array.min()).item()/measurement_size
        recovered_signal = generator(z[best, :].unsqueeze(0)).squeeze(0)

    return recovered_signal, np.array(losses), np.array(test_losses), smallest_loss, tol_stopped, last_iter

def make_df():
    """
    Create a DataFrame with experiment configurations for signal recovery.
    
    Returns:
        pd.DataFrame: DataFrame containing experiment parameters.
    """
    df = allcombinations({
        'lr': [1.0],  # Learning rate for optimization
        'beta1': [0.9],  # Adam optimizer parameter
        'beta2': [0.999],  # Adam optimizer parameter
        'max_iter': [3],  # Maximum number of iterations
        'opt_tol_stop': [1e-3],
        'num_repeats_in_optim': [100],  # Batch size for optimization (number of parallel attempts)
        'model_type': ['vae-32'],  # Model type for the generator
        'dataset': ['CELEBA'],  # Dataset used for signal generation
    })
    df = repeat(df, 6)
    global_seed = 1234
    np.random.seed(global_seed)
    df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
    df.attrs.update({
        'label': 'optimization_inversion',
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
    full_model = convvae.get_model(data['model_type']).eval()
    generator = VAEWrapper(full_model).to(device)

    def encoder(x):
        mu_p, log_var_p = full_model.forward_encoder(x)
        return full_model.sampling(mu_p, 2*log_var_p)

    # Generate a random target signal
    with torch.no_grad():
        # target_signal = generator(encoder(generator(torch.randn(1, generator.k).to(device)))).squeeze(0)
        target_code = torch.randn(1, generator.k).to(device)
        target_signal=generator(target_code).squeeze(0)

    # Recover the signal by inverting the generator
    recovered_signal, opt_data, test_losses, smallest_loss, tol_stopped, last_iter = exterior_redo_recover_signal(
        generator, target_signal, generator.k, lr = data['lr'], max_iter = data['max_iter'], 
        num_tries = data['num_repeats_in_optim'], tol_stop=data['opt_tol_stop'], 
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
        'experiment_fn_label': 'optimization_inversion',
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
