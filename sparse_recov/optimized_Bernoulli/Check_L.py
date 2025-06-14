# %%
from matplotlib.colors import LogNorm
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sparse_recov.sampling_recov_functions import Bernoulli_prob_weights
from utils.utils_data import logspace


def Lsr(j, m, n, sorted_alpha):
    if m >= n:
        raise ValueError("m cannot be greater or equal to n")
    return m * np.sum(sorted_alpha[:j]**2) / (m - (n-j))

# J = \max\left\{  j \in [n]: \frac{m \alpha_j^2}{L^2(j)} <  1\right\}


def get_J(m, n, sorted_alpha):
    if m >= n:
        raise ValueError("m cannot be greater than n")
    elif m <= 0:
        raise ValueError("m cannot be less than 0")
    for j in range(n, 1, -1):
        if m * sorted_alpha[j-1]**2 / Lsr(j, m, n, sorted_alpha) < 1:
            return j


def L(m, n, sorted_alpha):
    J = get_J(m, n, sorted_alpha)
    return Lsr(J, m, n, sorted_alpha)

# Create dataframe with m going from 1 to 100 as a column

# Define a function to calculate L for a given m


def get_calculate_L(n, sorted_alpha):
    return lambda m: L(m, n, sorted_alpha)

# %%
# Convert the above Julia code to Python

# def logspace(start, stop, length, base=2, integer=False):
#     if start <= 0 or stop <= 0:
#         raise ValueError("start and stop must be positive")
#     results = base ** np.linspace(np.log(start)/np.log(base), np.log(stop)/np.log(base), length)
#     if integer:
#         return np.round(results).astype(int)
#     return results


# %%
n = 10000
logspace(1, n-1, 1000, base=10, integer=True)


def plot_L(alpha):
    plt.figure(figsize=(10, 5))
    n = len(alpha)
    df = pd.DataFrame({'m': logspace(1, n-1, 1000, base=10, integer=True),
                      'L**2': 0.0, 'loc_co_norm_squared': np.sum(alpha**2), 'm/n': 0.0})
    # df['L**2'] = df['m'].apply(get_calculate_L(n, np.sort(alpha)[::-1]))
    df['L**2'] = df['m'].apply(lambda m: Bernoulli_prob_weights(alpha, m)[2])
    df['m/n'] = df.apply(lambda row: row['m'] / n, axis=1)
    df = df.melt(id_vars=['m/n'], value_vars=['L**2',
                 'loc_co_norm_squared'], value_name="Sample complexity factor")
    # rename column y to val
    sns.lineplot(x='m/n', y='Sample complexity factor',
                 data=df, hue='variable')
    # sns.lineplot(x='m', y='m', data=df, marker = 'o')
    plt.xscale('log')
    plt.yscale('log')
    # save as pdf plot

def get_flower_batch(num):
    # Realistic example
    transform = torchvision.transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    flowers = torchvision.datasets.Flowers102(
        "./datasets", split="test", transform=transform, download=False)
    dataloader = torch.utils.data.DataLoader(
        flowers, batch_size=num, shuffle=True)
    return next(iter(dataloader))[0]


one_batch = get_flower_batch(20)
one_batch.shape

def get_Fourier_co(outputs):
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

max_co = get_Fourier_co(one_batch)

ax = sns.heatmap(max_co[0,:,:], norm=LogNorm())
ax.set_xticks([])
ax.set_yticks([])
# %%

max_co_vec = max_co.flatten()
max_co_vec.shape

# perm = np.argsort(max_co_vec)
# sorted_max_co_vec = max_co_vec[perm]

plot_L(max_co_vec)
plt.ylabel("Sample complexity factor")
plt.title("Comparison of sample complexity factors")

# %%
# Examine explicitly the gain in sample complexity
# For m \geq L^2 D, Let 'D' be the isolated sample complexity


def plot_L_ratio(alpha):
    # TODO Fix the with replacement to go through the whole domain.
    n = len(alpha)
    plt.figure(figsize=(10, 5))
    df = pd.DataFrame({'m': logspace(1, n-1, 100, base=10, integer=True),
                      'L**2': 0.0, 'loc_co_norm_squared': np.sum(alpha**2)})
    df['L**2'] = df['m'].apply(lambda m:Bernoulli_prob_weights(alpha, m)[2])
    df['Bernoulli'] = df.apply(lambda row: row['m'] / row['L**2'], axis=1)
    df['With replacement'] = df.apply(lambda row: row['m'] /
                                                 row['loc_co_norm_squared'], axis=1)
    df = df.melt(id_vars=['m'], value_vars=[
                 'Bernoulli', 'With replacement'])
    max_common_elt = min(df.groupby("variable")['value'].max())
    df = df[(df['value']>=1)&(df['value']<=max_common_elt)]
    df['Sampling'] = df['variable']
    assert isinstance(df, pd.DataFrame)
    # Invert the function by permuting x and y axis.
    g = sns.lineplot(x='value', y='m', data=df, hue='Sampling')
    plt.xscale('log')
    plt.yscale('log')
    return g


plot_L_ratio(max_co_vec)
plt.xlabel("Sample complexity without the coherence factor")
plt.ylabel("Final sample complexity")
plt.title("Explicit sample complexity bound")
#plt.savefig("/Users/matthewscott/Obsidian/myVault/Files/Sample_complexity_improvement.pdf")

