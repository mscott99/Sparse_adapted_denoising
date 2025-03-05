import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sparse_recov.data_analysis import perform_least_squares_fit


def test_perform_least_squares_fit_basic():
    """Test basic functionality of perform_least_squares_fit with a simple dataset."""
    # Create synthetic data with known slope and intercept
    np.random.seed(42)
    x_values = np.logspace(0, 2, 20)  # 20 points from 10^0 to 10^2
    
    # Create data with slope = 2 and intercept = 1 in log space
    # y = 10^1 * x^2 = 10 * x^2
    y_values = 10 * x_values**2
    
    # Add some noise to make it realistic
    y_values_noisy = y_values * (1 + 0.1 * np.random.randn(len(y_values)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x_values,
        'y': y_values_noisy,
        'group': 'A'  # Single group
    })
    
    # Perform the fit
    results = perform_least_squares_fit(df, 'x', 'y', ['group'])
    
    # Check that we have one row in the results (one group)
    assert len(results) == 1
    
    # Check that the slope is close to 2 (since y = 10 * x^2)
    # In log space: log(y) = log(10) + 2*log(x)
    assert np.isclose(results.iloc[0]['slope'], 2.0, atol=0.1)
    
    # Check that the intercept is close to 1 (log(10) ≈ 1)
    assert np.isclose(results.iloc[0]['intercept'], 1.0, atol=0.2)
    
    # Check that r_squared is high (good fit)
    assert results.iloc[0]['r_squared'] > 0.95
    
    # Check that x_min and x_max are correct
    assert np.isclose(results.iloc[0]['x_min'], x_values.min())
    assert np.isclose(results.iloc[0]['x_max'], x_values.max())


def test_perform_least_squares_fit_multiple_groups():
    """Test perform_least_squares_fit with multiple groups."""
    np.random.seed(42)
    
    # Create data for two groups with different slopes
    x_values = np.logspace(0, 2, 30)
    
    # Group A: y = 10 * x^2
    y_values_A = 10 * x_values**2
    y_values_A_noisy = y_values_A * (1 + 0.05 * np.random.randn(len(y_values_A)))
    
    # Group B: y = 5 * x^1.5
    y_values_B = 5 * x_values**1.5
    y_values_B_noisy = y_values_B * (1 + 0.05 * np.random.randn(len(y_values_B)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': np.concatenate([x_values, x_values]),
        'y': np.concatenate([y_values_A_noisy, y_values_B_noisy]),
        'group': ['A'] * len(x_values) + ['B'] * len(x_values)
    })
    
    # Perform the fit
    results = perform_least_squares_fit(df, 'x', 'y', ['group'])
    
    # Check that we have two rows in the results (two groups)
    assert len(results) == 2
    
    # Get results for each group
    result_A = results[results['group'] == 'A'].iloc[0]
    result_B = results[results['group'] == 'B'].iloc[0]
    
    # Check slopes - group A should be close to 2, group B close to 1.5
    assert np.isclose(result_A['slope'], 2.0, atol=0.1)
    assert np.isclose(result_B['slope'], 1.5, atol=0.1)
    
    # Check that r_squared is high for both groups
    assert result_A['r_squared'] > 0.95
    assert result_B['r_squared'] > 0.95


def test_perform_least_squares_fit_with_min_x():
    """Test perform_least_squares_fit with min_x filter."""
    np.random.seed(42)
    
    # Create synthetic data
    x_values = np.logspace(0, 3, 50)  # 50 points from 10^0 to 10^3
    
    # Create data where the relationship changes at x = 100
    # For x < 100: y = x^1.2
    # For x >= 100: y = 5 * x^2
    y_values = np.where(
        x_values < 100, 
        x_values**1.2,
        5 * x_values**2
    )
    
    # Add noise
    y_values_noisy = y_values * (1 + 0.1 * np.random.randn(len(y_values)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x_values,
        'y': y_values_noisy,
        'group': 'A'
    })
    
    # Perform fit with min_x = 100 to focus on the second segment
    results = perform_least_squares_fit(df, 'x', 'y', ['group'], min_x=100)
    
    # Check that the slope is close to 2
    assert np.isclose(results.iloc[0]['slope'], 2.0, atol=0.1)
    
    # Check that x_min is approximately 100 (or the nearest value above 100)
    assert results.iloc[0]['x_min'] >= 100


def test_perform_least_squares_fit_multiple_group_cols():
    """Test perform_least_squares_fit with multiple grouping columns."""
    np.random.seed(42)
    
    # Create synthetic data
    x_values = np.logspace(0, 2, 40)
    
    # Create different slopes for different combinations of group1 and group2
    slopes = {
        ('A', 'X'): 1.5,  # group1=A, group2=X: y = x^1.5
        ('A', 'Y'): 2.0,  # group1=A, group2=Y: y = x^2
        ('B', 'X'): 1.0,  # group1=B, group2=X: y = x^1
        ('B', 'Y'): 2.5,  # group1=B, group2=Y: y = x^2.5
    }
    
    # Prepare data
    data = []
    for (group1, group2), slope in slopes.items():
        # Calculate y values based on the slope for this group
        y_values = 10 * x_values**slope
        y_values_noisy = y_values * (1 + 0.05 * np.random.randn(len(y_values)))
        
        # Add to data
        for x, y in zip(x_values, y_values_noisy):
            data.append({
                'x': x,
                'y': y,
                'group1': group1,
                'group2': group2
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Perform the fit
    results = perform_least_squares_fit(df, 'x', 'y', ['group1', 'group2'])
    
    # Check that we have four rows in the results (four group combinations)
    assert len(results) == 4
    
    # Check slopes for each group combination
    for (group1, group2), expected_slope in slopes.items():
        result_row = results[(results['group1'] == group1) & (results['group2'] == group2)].iloc[0]
        assert np.isclose(result_row['slope'], expected_slope, atol=0.1)


if __name__ == "__main__":
    # Run tests and print results
    pytest.main(["-v", __file__])