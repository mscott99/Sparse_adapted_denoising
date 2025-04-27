import pytest
import torch
import numpy as np
from sparse_recov.generative import Generator, generate_random_signal, get_fft_loc_coherences, CSMeasurementMatrix, recover_signal, make_df, experiment, sample_CS_matrix
from sparse_recov.train_model import ConvVAE
from sparse_recov.toy_Fourier import high_coherence_MNIST
from sparse_recov.sampling_recov_functions import half_robust
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# Fixture for a simple generator


@pytest.fixture
def generator():
    latent_dim = 10
    output_shape = (4, 4)  # Small size for quick testing
    return Generator(k=latent_dim, output_shape=output_shape)


def test_experiment():
    df = make_df()
    first_row = df.loc[0]
    first_row['max_iter'] = 2
    experiment({**df.attrs, **first_row})  # test for runtime errors


def test_run_batch():
    newdata = high_coherence_MNIST(l=2,perturbation_scale=3, data_dir="./data")
    loader = DataLoader(newdata)
    iter(loader).__next__()[0,0,:,:]
    plt.imshow(iter(loader).__next__()[0,0,:,:])
    plt.show()

def test_half_robust():
    """Test the half_robust function on a simple flattened coherence vector"""
    # Simple flattened coherence vector with values between 0 and 1
    loc_co = np.array([1.0, 0.5, 0.25, 0.125])

    # Get half-robust distribution
    p = half_robust(loc_co)

    # Calculate expected probabilities
    uniform_p = np.ones_like(loc_co) / len(loc_co)
    optimized_p = loc_co**2 / np.sum(loc_co**2)

    # Test if it's a probability distribution
    assert np.abs(np.sum(p) - 1.0) < 1e-10, "Does not sum to 1"
    assert np.all(p >= 0), "Contains negative probabilities"

    # Test edge case with zero coherence
    loc_co_zero = np.array([1.0, 0.5, 0.0, 0.0])
    p_zero = half_robust(loc_co_zero)
    assert np.all(
        p_zero >= 0), "Contains negative probabilities with zero coherence"
    assert np.abs(np.sum(p_zero) -
                  1.0) < 1e-10, "Does not sum to 1 with zero coherence"


def test_generator_integration():
    vae = ConvVAE(k=8)
    z = torch.randn(1, 8)
    output = vae.decode(z)
    assert output.shape == (
        1, 1, 28, 28), f"Expected output shape (1, 1, 28, 28), got {output.shape}"
    print("Generator integration test passed.")


def test_sample_CS_matrix_wr_optimized():
    """
    Test that sample_CS_matrix produces the correct preconditioning vector d
    for with-replacement sampling with optimized distribution on a 2D signal.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters for a 2D signal
    n1, n2 = 4, 4  # Small dimensions for easy verification
    n = n1 * n2  # Total number of elements (16)
    m = 8  # Number of measurements

    # Create a synthetic 2D local coherence array
    loc_co = np.ones((n1, n2))  # Base coherence of 1.0
    # Add variation for realism
    loc_co[0, 0] = 2.0  # Higher coherence at one point
    loc_co[1, 1] = 0.5  # Lower coherence at another point

    # Flatten loc_co for calculations
    loc_co_flat = loc_co.ravel()

    # Expected probabilities for optimized sampling
    p = loc_co_flat**2 / np.sum(loc_co_flat**2)
    # Expected d values: d[i] = 1 / sqrt(n * p[i]) where p[i] > 0, else 0
    expected_d = np.zeros_like(loc_co_flat)
    positive_p = p > 0
    expected_d[positive_p] = 1.0 / np.sqrt(n * p[positive_p])

    # Run the sampling function
    sampling_type = "wr"
    sampling_dist_type = "optimized"
    indices, d = sample_CS_matrix(loc_co, m, sampling_type, sampling_dist_type)

    # Basic checks
    assert isinstance(indices, np.ndarray), "Indices should be a numpy array"
    assert isinstance(
        d, np.ndarray), "Preconditioning vector d should be a numpy array"
    assert indices.size == m, f"Number of indices ({indices.size}) should equal m ({m})"
    assert d.shape == (
        n,), f"Preconditioning vector d should have shape ({n},)"
    assert np.all(indices >= 0) and np.all(
        indices < n), "Indices should be within [0, n)"
    assert np.all(np.isfinite(
        d)), "Preconditioning vector d should contain finite values"

    # Check that indices correspond to locations with positive probabilities
    assert np.all(
        p[indices] > 0), "Sampled indices should have positive probabilities"

    # Verify exact d values
    np.testing.assert_array_almost_equal(
        d, expected_d, decimal=10,
        err_msg="Preconditioning vector d does not match expected values"
    )

# Test generate_random_signal


def test_generate_random_signal(generator):
    latent_dim = generator.k
    device = 'cpu'
    signal = generate_random_signal(generator, latent_dim, device)

    # Check output type and shape
    assert isinstance(signal, torch.Tensor)
    assert signal.shape == (
        1, 1, *generator.output_shape), f"Expected shape (1, 1, {generator.output_shape}), got {signal.shape}"
    # Check values are within Hardsigmoid range [0, 1]
    assert torch.all(signal >= 0) and torch.all(
        signal <= 1), "Signal values should be in [0, 1]"

# Test get_fft_loc_coherences


def test_get_fft_loc_coherences(generator):
    num_samples = 5
    device = 'cpu'
    loc_co = get_fft_loc_coherences(
        generator, num_samples=num_samples, device=device)

    # Check output type and shape - get_fft_loc_coherences now removes the channel dimension
    assert loc_co.shape == generator.output_shape, f"Expected shape {generator.output_shape}, got {loc_co.shape}"
    # Check non-negative values (FFT magnitude)
    assert np.all(loc_co >= 0), "Local coherences should be non-negative"

# Test CSMeasurementMatrix forward


def test_cs_measurement_matrix_forward():
    signal_shape = (4, 4)
    n = np.prod(signal_shape)
    m = 8
    indices = torch.arange(0, n, n // m)  # Simple uniform sampling for test
    A = CSMeasurementMatrix(indices, m, signal_shape, preconditioned=False)

    # Create a dummy input
    x = torch.ones(2, *signal_shape)  # Batch of 2
    measurements = A(x)

    # Check output type and shape
    assert isinstance(measurements, torch.Tensor)
    assert measurements.shape == (2, len(
        indices)), f"Expected shape (2, {len(indices)}), got {measurements.shape}"
    assert measurements.is_complex(), "Measurements should be complex"

# Test recover_signal


def test_recover_signal(generator):
    latent_dim = generator.k
    signal_shape = generator.output_shape
    n = np.prod(signal_shape)
    m = 8
    indices = torch.arange(0, n, n // m)
    A = CSMeasurementMatrix(indices, m, signal_shape, preconditioned=False)

    # Generate a true signal and measurements
    true_signal = generate_random_signal(generator, latent_dim)
    b = A(true_signal)  # CSMeasurementMatrix now handles the channel dimension

    # Recover signal with lower iterations for test
    recovered_signal, _, _ = recover_signal(
        generator, A, b, latent_dim, lr=0.01, num_iterations=10, batch_size=2)

    # Check output type and shape - recover_signal will remove channel dimension
    assert isinstance(recovered_signal, torch.Tensor)
    assert recovered_signal.shape == signal_shape, f"Expected shape {signal_shape}, got {recovered_signal.shape}"
    # Check values are within Hardsigmoid range [0, 1]
    assert torch.all(recovered_signal >= 0) and torch.all(
        recovered_signal <= 1), "Recovered signal values should be in [0, 1]"
