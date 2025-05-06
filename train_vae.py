import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sparse_recov.train_model import ConvVAE
from sparse_recov.toy_Fourier import heavy_tail_MNIST_loaders
import pandas as pd
import seaborn as sns
from pathlib import Path

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Training hyperparameters
BATCH_SIZE = 128 
EPOCHS = 2
LEARNING_RATE = 1e-3
LATENT_DIM = 20
K = 3  # Number of FFT coefficients to perturb
PERTURB_SCALE = 5.0  # Scale of perturbation

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    recon_loss = 0 
    kl_loss = 0
    
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
            
        optimizer.zero_grad()
        recon_x, mu, log_var = model(x)
        
        # Calculate losses
        r_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        k_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss = r_loss + k_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        recon_loss += r_loss.item()
        kl_loss += k_loss.item()
        
    # Average losses over dataset
    n = len(loader.dataset)
    return total_loss/n, recon_loss/n, kl_loss/n

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    recon_loss = 0
    kl_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
                
            recon_x, mu, log_var = model(x)
            
            r_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
            k_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            total_loss += (r_loss + k_loss).item()
            recon_loss += r_loss.item()
            kl_loss += k_loss.item()
    
    n = len(loader.dataset)
    return total_loss/n, recon_loss/n, kl_loss/n

# Get MNIST datasets with heavy tail perturbation
train_dataset, test_dataset = heavy_tail_MNIST_loaders(
    k=K,
    perturbation_scale=PERTURB_SCALE,
    batch_size=BATCH_SIZE
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model and optimizer
model = ConvVAE(k=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize lists to store metrics
metrics = []

# Training loop
print("Starting training...")
for epoch in range(EPOCHS):
    train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer)
    test_loss, test_recon, test_kl = evaluate(model, test_loader)
    
    # Store metrics
    metrics.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_recon': train_recon,
        'train_kl': train_kl,
        'test_loss': test_loss,
        'test_recon': test_recon,
        'test_kl': test_kl
    })
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
        print(f"Test Loss: {test_loss:.4f} (Recon: {test_recon:.4f}, KL: {test_kl:.4f})")

# Convert metrics to DataFrame and save
metrics_df = pd.DataFrame(metrics)
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Save model and metrics
model_path = results_dir / f"vae_k{K}_scale{PERTURB_SCALE}_latent{LATENT_DIM}.pt"
metrics_path = results_dir / f"metrics_k{K}_scale{PERTURB_SCALE}_latent{LATENT_DIM}.csv"

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'hyperparameters': {
        'K': K,
        'PERTURB_SCALE': PERTURB_SCALE,
        'LATENT_DIM': LATENT_DIM,
        'LEARNING_RATE': LEARNING_RATE
    }
}, model_path)

print(f"Training complete. Model and metrics saved in {results_dir}")
