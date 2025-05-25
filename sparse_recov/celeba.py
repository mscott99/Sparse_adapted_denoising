from numpy._typing import NDArray
import numpy as np
from sparse_recov.convvae import VAE, get_model
import torch
import matplotlib.pyplot as plt

def generate_sample():
    model = get_model()
    model.decoder_conv
    model.z_dim
    # reconstructed = model.forward_decoder(z)

    # Generate a random sample from the latent space
    z = torch.randn(1, model.z_dim)
    # If model is on GPU, move z to GPU
    if next(model.parameters()).is_cuda:
        z = z.cuda()

    # Generate reconstruction from the latent sample
    with torch.no_grad():
        reconstructed = model.forward_decoder(z)

def plot_img(img: NDArray):
    # If image is in CHW format (channels, height, width), convert to HWC
    if img.shape[0] in [1, 3]:  # Assuming image might be 1 or 3 channels
        img = img.transpose(1, 2, 0)
    
    # If image values are in range [0,1], scale to [0,255]
    if img.max() <= 1.0:
        img = img * 255.0
    
    # Convert to uint8 if necessary
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    # If single channel, convert to 3 channels for display
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
        
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title('Reconstructed Image from VAE')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_pickle("./data/last_local.pickle")
    img = df["recovered_signal"][1]
    assert isinstance(img, np.ndarray)
    plot_img(img)
