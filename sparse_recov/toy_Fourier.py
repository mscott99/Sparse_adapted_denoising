import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


def perturb_image(images: torch.Tensor, l, scale: float):
    images_fft = torch.fft.fft2(images, dim=[-2, -1], norm="ortho")
    height, width = images.shape[2:]
    assert l <= height and l <= width

    perturbation = scale * \
        torch.randn(images_fft.shape[0], 1, 2*l-1,
                    2*l-1, dtype=torch.complex64)
    # Add perturbation to low frequencies
    images_fft[:, :, -l+1:, -l+1:] += perturbation[:, :, :l-1, :l-1]
    images_fft[:, :, :l, -l+1:] += perturbation[:, :, l-1:, :l-1]
    images_fft[:, :, -l+1:, :l] += perturbation[:, :, :l-1, l-1:]
    images_fft[:, :, :l, :l] += perturbation[:, :, l-1:, l-1:]
    perturbed_images = torch.fft.ifft2(
        images_fft, dim=(-2, -1), norm="ortho").real
    return torch.clip(perturbed_images, 0, 1)

# Create custom datasets from perturbed data


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def high_coherence_MNIST(l=5, perturbation_scale=0.1, data_dir='./data') -> CustomDataset:
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
    for images, _ in DataLoader(original_train_dataset, batch_size=128):
        perturbed_images = perturb_image(images, l, perturbation_scale)
        train_data_perturbed.append(perturbed_images)
    train_data_perturbed = torch.concat(train_data_perturbed, dim=0)
    return CustomDataset(train_data_perturbed)

# Create custom datasets from perturbed data (re-include for completeness)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Should be a tensor of images [N, C, H, W]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SparseFFTPerturbation:
    """
    A torchvision-compatible transform that adds exactly k complex Gaussian noise entries
    to the 2D FFT coefficients of an image tensor.

    Operates on a single image tensor (expected shape [C, H, W]).
    """

    def __init__(self, k: int, scale: float):
        """
        Args:
            k (int): Exact number of FFT coefficients to perturb.
            scale (float): Standard deviation of the complex Gaussian noise.
        """
        if k < 0:
            raise ValueError("k must be non-negative.")
        self.k = k
        self.scale = scale

    def _perturb_single_image_sparse(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Applies sparse FFT perturbation to a single image tensor [C, H, W]."""
        if image_tensor.ndim != 3:
            raise ValueError(
                f"Input tensor must be 3D ([C, H, W]). Got {image_tensor.ndim}D.")

        device = image_tensor.device
        images_fft = torch.fft.fft2(image_tensor, dim=[-2, -1], norm="ortho")

        # Total number of elements in FFT
        total_elements = np.prod(images_fft.shape)

        # Error check for k
        if self.k > total_elements:
            raise ValueError(
                f"k ({self.k}) cannot exceed total FFT elements ({total_elements})")

        # Create noise
        noise_real = torch.randn(images_fft.shape, device=device)
        noise_imag = torch.randn(images_fft.shape, device=device)
        complex_noise = self.scale * \
            (noise_real + 1j * noise_imag).to(dtype=torch.complex64)

        # Create mask with exactly k ones
        mask = torch.zeros(images_fft.shape, device=device, dtype=torch.bool)
        # Flatten indices, shuffle, and select first k
        flat_indices = torch.randperm(total_elements, device=device)[:self.k]
        # Convert to unflattened indices
        indices = np.unravel_index(
            flat_indices.cpu().numpy(), images_fft.shape)
        mask[indices] = True

        sparse_complex_noise = complex_noise * mask.to(complex_noise.dtype)
        images_fft_perturbed = images_fft + sparse_complex_noise
        perturbed_image = torch.fft.ifft2(
            images_fft_perturbed, dim=[-2, -1], norm="ortho").real
        return torch.clip(perturbed_image, 0, 1)

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the transform to an image tensor.

        Args:
            img_tensor (torch.Tensor): Input image tensor (expected [C, H, W]).

        Returns:
            torch.Tensor: Perturbed image tensor.
        """
        return self._perturb_single_image_sparse(img_tensor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k}, scale={self.scale})"


def heavy_tail_MNIST_loaders(
    k: int = 100,  # Number of FFT coefficients to perturb
    perturbation_scale: float = 0.5,
    data_dir='./data',
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False
) -> tuple[Dataset, Dataset]:
    """
    Creates MNIST DataLoaders where each image is dynamically perturbed
    with exactly k sparse FFT noise entries using a Dataset transform.

    Parameters:
    -----------
    k : int
        Exact number of FFT coefficients to perturb (default=100).
    perturbation_scale : float
        Scale (std dev) of the random complex Gaussian perturbation (default=0.5).
    data_dir : str
        Directory for MNIST dataset (default='./data').
    batch_size : int
        DataLoader batch size (default=64).
    num_workers : int
        DataLoader worker processes (default=0).
    pin_memory : bool
        DataLoader pin_memory option (default=False).

    Returns:
    --------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the perturbed training dataset.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the perturbed test dataset.
    """

    heavy_tail_transform = transforms.Compose([
        transforms.ToTensor(),  # MUST come first
        SparseFFTPerturbation(k=k, scale=perturbation_scale)
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=heavy_tail_transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=heavy_tail_transform
    )

    loading_device = "CPU (default)"
    if num_workers == 0:
        loading_device = "CPU (main process)"
    print(
        f"Perturbation transform will be applied on: {loading_device} by DataLoader workers")

    return train_dataset, test_dataset
