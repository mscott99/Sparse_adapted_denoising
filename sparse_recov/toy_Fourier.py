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
        self.data = data # Should be a tensor of images [N, C, H, W]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SparseFFTPerturbation:
    """
    A torchvision-compatible transform that adds sparse complex Gaussian noise
    to the 2D FFT coefficients of an image tensor.

    Operates on a single image tensor (expected shape [C, H, W]).
    """
    def __init__(self, density: float, scale: float):
        """
        Args:
            density (float): Fraction of FFT coefficients to perturb (0 to 1).
            scale (float): Standard deviation of the complex Gaussian noise.
        """
        if not 0.0 <= density <= 1.0:
            raise ValueError("Density must be between 0 and 1.")
        self.density = density
        self.scale = scale

    def _perturb_single_image_sparse(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Applies sparse FFT perturbation to a single image tensor [C, H, W]."""
        if image_tensor.ndim != 3:
             raise ValueError(f"Input tensor must be 3D ([C, H, W]). Got {image_tensor.ndim}D.")
        device = image_tensor.device # Perform calculations on the tensor's device
        images_fft = torch.fft.fft2(image_tensor, dim=[-2, -1], norm="ortho")
        noise_real = torch.randn(images_fft.shape, device=device)
        noise_imag = torch.randn(images_fft.shape, device=device)
        complex_noise = self.scale * (noise_real + 1j * noise_imag).to(dtype=torch.complex64)
        mask = torch.rand(images_fft.shape, device=device) < self.density
        sparse_complex_noise = complex_noise * mask.to(complex_noise.dtype)
        images_fft_perturbed = images_fft + sparse_complex_noise
        perturbed_image = torch.fft.ifft2(images_fft_perturbed, dim=[-2, -1], norm="ortho").real
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
        return f"{self.__class__.__name__}(density={self.density}, scale={self.scale})"

def heavy_tail_MNIST_loaders(
    perturbation_density: float = 0.01,
    perturbation_scale: float = 0.5,
    data_dir='./data',
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False
    ) -> tuple[DataLoader, DataLoader]:
    """
    Creates MNIST DataLoaders where each image is dynamically perturbed
    with sparse FFT noise using a Dataset transform.

    Parameters:
    -----------
    perturbation_density : float
        Fraction of FFT coefficients to perturb (0 to 1, default=0.01).
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
        transforms.ToTensor(), # MUST come first
        SparseFFTPerturbation(density=perturbation_density, scale=perturbation_scale)
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=heavy_tail_transform # Apply our transform here
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=heavy_tail_transform # Apply to test set too
    )

    loading_device = "CPU (default)" # Transforms in workers run on CPU
    if num_workers == 0:
       loading_device = "CPU (main process)"
    print(f"Perturbation transform will be applied on: {loading_device} by DataLoader workers")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Created DataLoaders with dynamic sparse perturbation transform (density={perturbation_density}, scale={perturbation_scale}).")
    return train_loader, test_loader


if __name__ == '__main__':
    BATCH_SIZE = 128
    PERTURB_DENSITY = 0.05 # Perturb 2% of coefficients
    PERTURB_SCALE = 1.0   # Slightly stronger noise

    print("Creating DataLoaders with dynamic heavy-tail perturbation via Dataset transform...")
    ht_train_loader_transform, ht_test_loader_transform = heavy_tail_MNIST_loaders(
        perturbation_density=PERTURB_DENSITY,
        perturbation_scale=PERTURB_SCALE,
        batch_size=BATCH_SIZE,
        num_workers=2, # Use 2 workers for parallel loading & transform
        pin_memory=torch.cuda.is_available()
    )
    print("DataLoaders created.")

    # Get one batch from the training loader to demonstrate
    print("\nFetching one batch from the training loader (using transform)...")
    try:
        first_batch_images_t, first_batch_labels_t = next(iter(ht_train_loader_transform))

        print(f"Fetched batch shapes:")
        print(f"  Images: {first_batch_images_t.shape}")
        print(f"  Labels: {first_batch_labels_t.shape}")
        # Device depends on pin_memory and main process/worker
        print(f"  Images device: {first_batch_images_t.device}")
        print(f"  Labels device: {first_batch_labels_t.device}")

        # Visualize a few images from the batch
        import matplotlib.pyplot as plt

        num_images_to_show = 5
        plt.figure(figsize=(10, 3))
        plt.suptitle(f"Sample Perturbed MNIST (Transform) (Density={PERTURB_DENSITY}, Scale={PERTURB_SCALE})")
        for i in range(num_images_to_show):
            if i >= first_batch_images_t.shape[0]: break
            ax = plt.subplot(1, num_images_to_show, i + 1)
            img_to_show = first_batch_images_t[i].squeeze().cpu().numpy() # Move to CPU for numpy/plotting
            plt.imshow(img_to_show, cmap='gray')
            plt.title(f"Label: {first_batch_labels_t[i].item()}")
            plt.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"Error fetching or visualizing batch: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure you have matplotlib installed (`pip install matplotlib`)")
