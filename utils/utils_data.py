import inspect
import os
import subprocess
import traceback
import pandas as pd
from tqdm import tqdm
import itertools
import datetime
import numpy as np
from typing import Any, Union, Optional
from joblib import Parallel, delayed

def soft_astype_df(df:pd.DataFrame, dtype_dict:dict[str, Any]):
    valid_dtypes = {col: dtype_dict[col] for col in df.columns if col in dtype_dict}
    df = df.astype(valid_dtypes)
    return df

def melt_and_keep(df:pd.DataFrame, value_vars, var_name='variable', value_name='value'):
    id_vars = [col for col in df.columns if col not in value_vars]
    melted_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    return melted_df

def logspace(start, stop, length, base=2, integer=True):
    if start <= 0 or stop <= 0:
        raise ValueError("start and stop must be positive")
    results = base ** np.linspace(np.log(start)/np.log(base), np.log(stop)/np.log(base), length)
    if integer:
        return np.round(results).astype(int)
    return results

def allcombinations(mydict:dict, dtypes_:Union[None, dict[str, Any]] =None) -> pd.DataFrame:
    # mydict = a
    for (key, value) in mydict.items():
        if not isinstance(value, (list, tuple, set, pd.Series, np.ndarray)):
            mydict[key] = [value]  # Wrap in a list
    combinations = list(itertools.product(*mydict.values()))
    df = pd.DataFrame.from_records(combinations, columns=mydict.keys())
    if dtypes_ == None:
        return df
    return df.astype(dtypes_)

def economical_df(base_dict: dict, auxiliarydict: dict, dtypes_: Union[None, dict[str, Any]] = None) -> pd.DataFrame:
    """
    Auxiliary dict should have:
    {
    key: (default_value, [values_list])
    }
    """
    center_dict = base_dict.copy()
    for aux in auxiliarydict.keys():
        center_dict[aux] = auxiliarydict[aux][0]

    ref_df = allcombinations(center_dict)
    dfs = []
    for aux_col, aux_values in auxiliarydict.items():
        # Remove the reference value for current auxiliary column
        current_dict = base_dict.copy()
        current_dict[aux_col] = aux_values[1]
        for other_aux in auxiliarydict.keys():
            if other_aux != aux_col:
                current_dict[other_aux] = auxiliarydict[other_aux][0]
            
        df = allcombinations(current_dict)
        dfs.append(df)
    
    if dfs:
        result = pd.concat([ref_df] + dfs, axis=0, ignore_index=True)
    else:
        result = ref_df
    if dtypes_ is not None:
        result = soft_astype_df(result, dtypes_)
    return result

def repeat(df: pd.DataFrame, times ) -> pd.DataFrame:
    return pd.concat([df]*times, ignore_index=True)

def cross_merge_dfs(df1, df2):
    return pd.merge(df1, df2, how='cross')

def run_local_df_experiment(df:pd.DataFrame, fn:Any, dtype:Union[dict[str, Any], None]=None, catch_errors=False)-> pd.DataFrame:
    """
    Run the df method for a df that possibly has attributes.
    Args:
        df: Dataframe with the experiment data.
        fn: The row-wise function, should return a pandas series with the value of the new columns.
        dtype: Optional dictionary specifying data types for the resulting columns.
    Returns:
        pd.DataFrame: The extended dataframe with results from the function.
    """
    total_rows = len(df)
    processed_rows = []
    
    def safe_exp(data):
        try:
            return fn(data)
        except Exception:
            return {"ERROR": traceback.format_exc()}
    # Create a progress bar with tqdm
    with tqdm(total=total_rows, desc="Processing rows", unit="rows") as pbar:
        for idx, row in enumerate(df.itertuples(), 1):
            # Process row
            row_dict = row._asdict()
            del row_dict['Index']  # Remove the Index field from namedtuple
            if catch_errors:
                result = pd.Series(safe_exp({**df.attrs, **row_dict}))
            else:
                result = pd.Series(fn({**df.attrs, **row_dict}))

            processed_rows.append(result)
            
            # Update progress bar
            pbar.update(1)
    
    # Combine results
    result = pd.concat([df, pd.DataFrame(processed_rows)], axis=1)
    
    if dtype is not None:
        return soft_astype_df(result, dtype)
    return result

def add_standard_attributes(df:pd.DataFrame):
    df.attrs.update({'global_seed':1234, 'datetime':datetime.datetime.now().strftime('%Y-%m-%d')})

def get_git_commit_hash(func) -> Union[str, None]:
    """
    Finds the Git commit hash for the repository where a function is defined.

    Args:
        func (function): The function whose repository's commit hash is needed.

    Returns:
        str | None: The Git commit hash (short form) as a string, or None if it cannot be determined
                    (e.g., not in a Git repo, or git command fails).
    """
    try:
        func_file_path = inspect.getfile(func)
        repo_dir = os.path.dirname(func_file_path)
        if not os.path.isdir(repo_dir):
            print("Warning: Could not determine function's directory.")
            return None
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (TypeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not get git commit hash. Reason: {e}")
        return None


def run_local_parallel_experiment(df: pd.DataFrame, fn, dtype: Optional[dict] = None, random_state=42) -> pd.DataFrame:
    """
    Run a function in parallel on each row of a DataFrame, with random shuffle for load balancing,
    while ensuring results align with their corresponding input rows. The git commit hash of the
    function `fn` is automatically added to the DataFrame's attributes.

    Args:
        df (pd.DataFrame): DataFrame with the experiment data. Can have .attrs.
        fn (function): The row-wise function. It receives a dictionary combining the row data and df.attrs.
            It should return a dictionary or pandas.Series with the new column values.
        dtype (dict, optional): Dictionary specifying data types for the resulting columns. Defaults to None.
        random_state (int, optional): Seed for the random shuffle for reproducibility. 
            Set to None for a non-deterministic shuffle. Defaults to 42.

    Returns:
        pd.DataFrame: The original dataframe extended with the new result columns, in the original row order.
    """
    original_attrs = df.attrs.copy()
    print("Number of tasks: ", len(df))
    commit_hash = get_git_commit_hash(fn)
    if commit_hash:
        original_attrs['fn_commit_hash'] = commit_hash
    shuffled_df = df.sample(frac=1, random_state=random_state)

    def worker(row: pd.Series):
        """
        Processes a single row. The row's 'name' attribute holds its original index.
        Returns a tuple of (original_index, result_data).
        """
        combined_data = {**original_attrs, **row.to_dict()}
        try:
            result = fn(combined_data)
        except Exception:
            result = {"ERROR": traceback.format_exc()}
        return (row.name, result)

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(worker)(row) for _, row in shuffled_df.iterrows()
    )
    if not results:
        df.attrs.update(original_attrs)
        return df
    result_indices, result_data = zip(*results)
    result_df = pd.DataFrame(list(result_data))
    result_df.index = result_indices
    final_df = df.join(result_df)
    final_df.attrs = original_attrs
    if dtype:
        final_df = soft_astype_df(final_df, dtype)
    return final_df
