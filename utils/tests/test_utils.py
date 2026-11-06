import pytest
import pandas as pd
import numpy as np
from utils.utils_data import run_local_df_experiment, economical_df, run_local_parallel_experiment

def test_economical_df():
    # Test case 1: Basic functionality
    core_dict = {
        'A': [1, 2],
    }
    auxiliary_dict = {
        'B': ('x', ['y', 'z']),
        'C': (True, [False])
    }
    result = economical_df(core_dict, auxiliary_dict)
    print(economical_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 8  # 2*1 (reference) + 2*2 (B='z') + 2*1 (C=False)
    
    # Test case 2: Empty auxiliary dictionary
    result_empty = economical_df(core_dict, {})
    assert len(result_empty) == 2 
    
    # Test case 3: Test with dtypes
    dtypes = {'A': int, 'B': str, 'C': bool}
    result_typed = economical_df(core_dict, auxiliary_dict, dtypes_=dtypes)
    assert result_typed['A'].dtype == np.int64
    assert result_typed['B'].dtype == object
    assert result_typed['C'].dtype == bool
    
    # Test case 5: Test when auxiliary values are same as core
    aux_dict_same = {'B': ['x', 'y']}
    result_same = economical_df(core_dict, aux_dict_same)
    assert len(result_same) == 4  # Should be same as core combinations

def test_run_local_df_experiment():
    # Simple dataframe with 3 rows
    df = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [4, 5, 6]
    })
    
    # Simple function that squares x and adds y
    def fn(row):
        return {'result': row['x']**2 + row['y']}
    
    result = run_local_parallel_experiment(df, fn)
    
    # Check output format and values
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert 'result' in result.columns
    assert result['result'].tolist() == [5, 9, 15]  # 1²+4, 2²+5, 3²+6
