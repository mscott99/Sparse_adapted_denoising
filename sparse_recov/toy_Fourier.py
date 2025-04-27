import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


def perturb_image(images:torch.Tensor, l, scale:float):
    images_fft = torch.fft.fft2(images, dim=[-2,-1], norm="ortho")
    height, width = images.shape[2:]
    assert l <= height and l <= width
    
    perturbation = scale * torch.randn(images_fft.shape[0], 1, 2*l-1, 2*l-1, dtype=torch.complex64)
    # Add perturbation to low frequencies
    images_fft[:, :, -l+1:, -l+1:] += perturbation[:,:,:l-1, :l-1]
    images_fft[:, :, :l, -l+1:] += perturbation[:,:,l-1:,:l-1]
    images_fft[:, :, -l+1:, :l] += perturbation[:,:,:l-1, l-1:]
    images_fft[:, :, :l, :l] += perturbation[:,:,l-1:,l-1:]
    perturbed_images = torch.fft.ifft2(images_fft, dim=(-2,-1), norm="ortho").real
    return torch.clip(perturbed_images, 0, 1)

# Create custom datasets from perturbed data
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def high_coherence_MNIST(l=5, perturbation_scale=0.1, data_dir='./data', batch_size=64) -> CustomDataset:
    """
    Modifies the MNIST dataset by adding perturbations in the first l low-frequency
    directions of the 2D FFT to create high-coherence data. Returns DataLoader objects.

    Parameters:
    -----------
    l : int
        Number of low-frequency directions to perturb (default=5).
        Should be less than or equal to the minimum dimension of the image.
    perturbation_scale : float
        Scale of the random perturbation to be added (default=0.1).
        Controls the magnitude of the added noise.
    data_dir : str
        Directory to download/store the MNIST dataset (default='./data').
    batch_size : int
        Batch size for the DataLoader (default=64).

    Returns:
    --------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the perturbed training dataset.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the perturbed test dataset.
    original_train_loader : torch.utils.data.DataLoader
        DataLoader for the original (unperturbed) training dataset (for comparison).
    """
    base_transform = transforms.Compose([
        transforms.ToTensor()  # Converts to tensor and normalizes to [0, 1]
    ])
    original_train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=base_transform
    )
    train_data_perturbed = []
    train_labels = []
    for images, label in DataLoader(original_train_dataset, batch_size=128):
        perturbed_images = perturb_image(images, l, perturbation_scale)
        train_data_perturbed.append(perturbed_images)
    train_data_perturbed = torch.concat(train_data_perturbed, dim=0)


    return CustomDataset(train_data_perturbed)


# Example usage and plotting when script is run directly
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(43)
    original_train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=False, transform=transforms.ToTensor()
    )
    example_indices = [0, 1, 2, 3]  # Use first 4 images
    fig, axes = plt.subplots(2, len(example_indices), figsize=(15, 6))
    original_images = torch.stack(
        [original_train_dataset[i][0] for i in example_indices])
    perturbed = perturb_image(original_images, l=3, scale=2)
    # Plot original and perturbed images
    for i in range(len(example_indices)):
        # Plot original image in top row
        axes[0, i].imshow(original_images[i, 0, :, :].numpy(), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')

        # Plot perturbed image in bottom row
        axes[1, i].imshow(perturbed[i, 0, :, :].numpy(), cmap='gray')
        axes[1, i].set_title(f'Perturbed {i+1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
