import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple, Callable
import numpy as np

class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for MNIST.
    Encoder uses convolutions to map 28x28 images to latent space,
    Decoder uses transposed convolutions to map latent space to images.
    """
    def __init__(self, latent_dim: int):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),  # 64 * 7 * 7 = 3136
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )
        
        # Latent space mean and log-variance
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: Fully connected to reshape, then transposed convolutions
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),  # Reshape to (64, 7, 7)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()  # Output in [0,1] for MNIST
        )
    
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
        h = self.decoder_fc(z)  # Map latent to feature space
        recon_x = self.decoder(h)  # Reconstruct image
        return recon_x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Reconstruction loss (binary cross-entropy)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
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
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}")

def get_trained_generator(latent_dim: int, signal_dim: int = 784) -> Tuple[nn.Module, Callable]:
    """
    Train and return the decoder of a convolutional VAE as a generative model.
    
    Args:
        latent_dim: Dimension of the latent space (Gaussian N(0,1))
        signal_dim: Dimension of the output signal (784 for MNIST, ignored here as shape is fixed)
    
    Returns:
        decoder: Trained convolutional decoder (generator)
        reshape_fn: Function to reshape flat signal to 2D (28x28)
    """
    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                              transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize convolutional VAE
    vae = ConvVAE(latent_dim=latent_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train the VAE
    train_vae(vae, train_loader, num_epochs=20, device=device)
    
    # Extract the decoder as the generator
    def decoder(z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return vae.decode(z)
    
    # Define reshape function for 28x28 images
    def reshape_fn(x: np.ndarray) -> np.ndarray:
        return x.reshape(28, 28)
    
    return decoder, reshape_fn

# Example usage
if __name__ == "__main__":
    latent_dim = 100
    decoder, reshape_fn = get_trained_generator(latent_dim)
    
    # Generate a sample image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)  # Sample from N(0,1)
        img_tensor = decoder(z).cpu().numpy().squeeze()  # Shape: (1, 28, 28) -> (28, 28)
    
    # Optionally visualize
    import matplotlib.pyplot as plt
    plt.imshow(img_tensor, cmap='gray')
    plt.show()
