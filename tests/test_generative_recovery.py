import pytest
import torch
import numpy as np
from sparse_recov.generative import Generator, generate_random_signal, get_fft_loc_coherences, CSMeasurementMatrix, recover_signal, make_df, experiment

# Fixture for a simple generator
@pytest.fixture
def generator():
    latent_dim = 10
    output_shape = (4, 4)  # Small size for quick testing
    return Generator(latent_dim=latent_dim, output_shape=output_shape)

def test_experiment():
    df = make_df()
    first_row = df.loc[0]
    first_row['max_iter'] = 2
    experiment({**df.attrs, **first_row})  # test for runtime errors

# Test generate_random_signal
def test_generate_random_signal(generator):
    latent_dim = generator.k
    device = 'cpu'
    signal = generate_random_signal(generator, latent_dim, device)
    
    # Check output type and shape
    assert isinstance(signal, torch.Tensor)
    assert signal.shape == (1, *generator.output_shape), f"Expected shape (1, {generator.output_shape}), got {signal.shape}"
    # Check values are within Hardsigmoid range [0, 1]
    assert torch.all(signal >= 0) and torch.all(signal <= 1), "Signal values should be in [0, 1]"

# Test get_fft_loc_coherences
def test_get_fft_loc_coherences(generator):
    num_samples = 5
    device = 'cpu'
    loc_co = get_fft_loc_coherences(generator, num_samples=num_samples, device=device)
    
    # Check output type and shape
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
    assert measurements.shape == (2, len(indices)), f"Expected shape (2, {len(indices)}), got {measurements.shape}"
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
    b = A(true_signal)
    
    # Recover signal
    recovered_signal = recover_signal(generator, A, b, latent_dim, lr=0.01, num_iterations=10, batch_size=2)
    
    # Check output type and shape
    assert isinstance(recovered_signal, torch.Tensor)
    assert recovered_signal.shape == signal_shape, f"Expected shape {signal_shape}, got {recovered_signal.shape}"
    # Check values are within Hardsigmoid range [0, 1]
    assert torch.all(recovered_signal >= 0) and torch.all(recovered_signal <= 1), "Recovered signal values should be in [0, 1]"
