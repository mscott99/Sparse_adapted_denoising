from seaborn import heatmap
import torch
import numpy as np
from .generative import get_fft_loc_coherences
from .train_model import ConvGenerator
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


def plot_co(G:ConvGenerator):
    heatmap(np.fft.fftshift(get_fft_loc_coherences(G)[0,:,:]), norm=LogNorm())

def show_images(G: ConvGenerator, num_images: int = 4, device: str = 'cpu'):
    # Set the generator to evaluation mode
    G.eval()
    G.to(device)
    
    # Generate random latent vectors from a standard normal distribution
    z = torch.randn(num_images, G.k).to(device)
    
    # Generate images from the latent vectors
    with torch.no_grad():
        generated_images = G(z)  # Shape: (num_images, 1, 28, 28)
    
    # Create a figure for plotting with adjusted size
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    
    # Plot each generated image
    for i in range(num_images):
        axes[i].imshow(generated_images[i, 0, :, :].cpu().numpy(), cmap='gray')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
