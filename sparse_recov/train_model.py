import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple, Callable
import numpy as np
import pickle
from sparse_recov.toy_Fourier import high_coherence_MNIST, heavy_tail_MNIST_loaders
from abc import ABC, abstractmethod
from .convvae import VAE

class Generator(nn.Module, ABC):
    k: int
    output_shape: Tuple
    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
       """Decode a code to an image."""
       pass

class VAEWrapper(Generator):
    def __init__(self, vae:VAE):
        super().__init__()
        vae.eval()
        self.decoder_linear = vae.decoder_linear
        self.decoder_conv = vae.decoder_conv
        self.k = vae.z_dim
        self.conv_out_shape = vae.conv_out_shape
        self.output_shape = vae.input_shape

    def forward(self, z:torch.Tensor)-> torch.Tensor:
        z = self.decoder_linear(z)
        z = z.view(z.size()[0], *self.conv_out_shape[1:])
        z = self.decoder_conv(z)
        return z

class ConvGenerator(Generator):
    """
    A generative neural network using transpose convolutions for compressed sensing.
    Maps a latent Gaussian vector to a signal of shape (batch_size, 1, 28, 28).
    """
    k: int
    output_shape: Tuple

    def __init__(self, k: int, output_shape: Tuple, hidden_channels: list = [64, 32, 1]):
        super(ConvGenerator, self).__init__()
        self.k = k
        self.output_shape = output_shape

        # Project latent vector to a small spatial feature map (64 channels, 7x7)
        self.proj = nn.Linear(k, 64 * 7 * 7)

        # Transpose convolution layers
        self.network = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4,
                               stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2,
                               padding=1),   # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (batch_size, latent_dim)
        x = self.proj(z)  # (batch_size, 64 * 7 * 7)
        x = x.view(-1, 64, 7, 7)  # (batch_size, 64, 7, 7)
        x = self.network(x)  # (batch_size, 1, 28, 28)
        return x


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for MNIST.
    Encoder uses convolutions to map 28x28 images to latent space,
    Decoder uses the Generator class to map latent space to images.
    """
    k: int

    def __init__(self, k: int = 8):
        super(ConvVAE, self).__init__()
        self.k = k
        self.signal_shape = (1, 28, 28)

        # Encoder: Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2,
                      padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2,
                      padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),  # 64 * 7 * 7 = 3136
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )

        # Latent space mean and log-variance
        self.fc_mu = nn.Linear(256, k)
        self.fc_logvar = nn.Linear(256, k)

        # Decoder: Use Generator class
        self.decoder = ConvGenerator(k=self.k, output_shape=(
            28, 28), hidden_channels=[256, 512, 1024])

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)  # Input: (batch, 1, 28, 28)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Reconstruction loss (binary cross-entropy)
    recon_loss = nn.functional.binary_cross_entropy(
        recon_x, x, reduction='sum')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss


def get_fft_co_with_grad(generator: ConvGenerator, num_samples: int = 15, device: str = 'cpu') -> torch.Tensor:
    codes = torch.randn(num_samples, generator.k,
                        requires_grad=False).to(device)
    outputs = generator(codes)  # Shape: (num_samples, 1, 28, 28)
    outputs_i = outputs.unsqueeze(1)  # (num_samples, 1, 28, 28)
    outputs_j = outputs.unsqueeze(0)  # (1, num_samples, 28, 28)
    # (num_samples, num_samples, 28, 28)
    differences = outputs_i - outputs_j
    norms = torch.norm(differences,p=2, dim=(2, 3, 4), keepdim=True)
    normalized_diffs = differences / \
        (norms + 1e-8)  # Avoid division by zero

    # Step 5: Compute FFT of the normalized differences
    # Convert to complex for FFT
    coefs = torch.fft.fft2(normalized_diffs, norm="ortho")
    loc_co = torch.max(torch.abs(coefs), dim=0)[0]  # (num_samples^2, 28, 28)
    loc_co = torch.max(loc_co, dim=0)[0]
    return loc_co  # Return as numpy array


def train_vae(vae: ConvVAE, train_loader: DataLoader, epochs: int = 20,
              lr: float = 1e-3, co_param=0.0, device: str = 'cpu', **kwargs) -> None:
    """
    Train the convolutional VAE on MNIST.
    """
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(data)
            v_loss = vae_loss(recon_batch, data, mu, logvar)

            co_loss = -torch.sum(get_fft_co_with_grad(vae.decoder, 15))
            loss = v_loss + co_param*co_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}")


def get_trained_generator(latent_dim: int, train_dataset, co_param=0.0, **kwargs) -> nn.Module:
    """
    Train and return the Generator from a convolutional VAE as a generative model.

    Args:
        latent_dim: Dimension of the latent space (Gaussian N(0,1))
        signal_dim: Dimension of the output signal (784 for MNIST, ignored here as shape is fixed)

    Returns:
        decoder: Trained Generator instance
        reshape_fn: Function to reshape flat signal to 2D (28x28)
    """
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Initialize convolutional VAE
    vae = ConvVAE(k=latent_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train the VAE
    train_vae(vae, train_loader, co_param=co_param, device=device, **kwargs)

    return vae

def vanilla_train_vae(label="MNIST_VAE_"):
    latent_dim = 16
    vae = ConvVAE(k=latent_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transforms.ToTensor())
    vae = get_trained_generator(
            latent_dim, train_dataset)
    torch.save(vae.state_dict(), f"./models/{label}-{latent_dim}.pth")
    

def train_with_heavy_tail_data():
    latent_dim = 16
    # Load MNIST dataset
    # transform = transforms.ToTensor()
    # train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
    # train_dataset = high_coherence_MNIST(l=2, perturbation_scale=4, batch_size=512)# nine perturbed dimensions
    train_dataset, _ = heavy_tail_MNIST_loaders(0.05, 1.0)
    vae = get_trained_generator(latent_dim, train_dataset)
    torch.save(vae.state_dict(), 'heavy_tail_v0.pth')


def train_with_heavy_tail_loss_co(co_param=0.5, label = "model", **kwargs):
    latent_dim = 16
    # Load MNIST dataset
    # transform = transforms.ToTensor()
    # train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
    # train_dataset = high_coherence_MNIST(l=2, perturbation_scale=4, batch_size=512)# nine perturbed dimensions

    transf = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transf
    )
    vae = get_trained_generator(
        latent_dim, train_dataset, co_param=co_param, **kwargs)
    torch.save(vae.state_dict(), f"{label}.pth")
