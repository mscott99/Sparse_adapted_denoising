import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple, Callable
import numpy as np
import pickle
from sparse_recov.toy_Fourier import high_coherence_MNIST


class Generator(nn.Module):
    """
    A generative neural network using transpose convolutions for compressed sensing.
    Maps a latent Gaussian vector to a signal of shape (batch_size, 1, 28, 28).
    """
    k: int
    output_shape: Tuple

    def __init__(self, k: int, output_shape: Tuple, hidden_channels: list = [64, 32, 1]):
        super(Generator, self).__init__()
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
    def __init__(self, k: int=8):
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
        self.decoder = Generator(k=self.k, output_shape=(
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


def train_vae(vae: ConvVAE, train_loader: DataLoader, num_epochs: int = 20,
              lr: float = 1e-3, device: str = 'cpu') -> None:
    """
    Train the convolutional VAE on MNIST.
    """
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}")


def get_trained_generator(latent_dim: int, train_dataset) -> nn.Module:
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
    train_vae(vae, train_loader, num_epochs=60, device=device)

    return vae


# Example usage
if __name__ == "__main__":
    latent_dim = 16
    # Load MNIST dataset
    # transform = transforms.ToTensor()
    # train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
    train_dataset = high_coherence_MNIST(l=2, perturbation_scale=3, batch_size=512)# nine perturbed dimensions
    vae = get_trained_generator(latent_dim, train_dataset)
    torch.save(vae.state_dict(), 'perturbed_model_weights_v3.pth')

