import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal
from matplotlib.colors import LogNorm

import seaborn as sns
import numpy as np


def log_relplot(data, x, y, hue=None, col=None, row=None, style=None, size=None, ax=None, logx=True, logy=True, estimator='gmean', errorbar='gse', palette="plasma", hue_norm=None, kind:Literal["scatter", "line"] = "line", **kwargs):
    """
    A wrapper around seaborn.relplot with log-log scales and geometric means/errors by default.

    Parameters:
    - data: DataFrame, input data.
    - x, y: str, column names for x and y axes.
    - hue, col, row, style, size: str or None, categorical mappings (same as relplot).
    - logx, logy: bool, whether to use log scale for x/y axes (True by default).
    - get_errorbar: str or callable, error type ('gse' for geometric SE by default, or Seaborn options).
    - **kwargs: Additional arguments passed to seaborn.relplot.

    Returns:
    - Seaborn FacetGrid object.
    """
    # Define geometric mean and geometric SE calculation
    def geo_mean(x): return np.exp(np.mean(np.log(x)))

    def geo_se(x):
        # SE of log-transformed data
        s = np.std(np.log(x), ddof=1) / np.sqrt(len(x))
        gm = geo_mean(x)
        return [gm * np.exp(-s), gm * np.exp(s)]  # Absolute min/max bounds

    # Handle 'gse' error type
    if errorbar == 'gse':
        def get_errorbar(x): return geo_se(x) if len(x) > 1 else [
            x[0], x[0]]  # Avoid NaN, no range for n=1
        errorbar = get_errorbar

    if estimator == "gmean":
        estimator = geo_mean

    if hue is not None and pd.api.types.is_number(data[hue]) and not  hue_norm is None:
        print("went ahead")
        hues = data[hue]
        # Filter positive values for log norm
        hues_non_zero = hues[hues > 0]
        if len(hues_non_zero) > 0:  # Check if we have any positive values
            if 0.0 not in hues:
                hue_norm = LogNorm(vmin=hues.min(), vmax=hues.max(), clip=True)
            else:
                hue_norm = LogNorm(
                    vmin=np.exp(np.log(hues_non_zero.min()) - 0.3*(np.log(hues_non_zero.max()) - np.log(hues_non_zero.min()))), vmax=hues_non_zero.max(), clip=True)

    if row is None and col is None and ax is not None:
        if kind == "line":
            g = sns.lineplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                style=style,
                hue_norm=hue_norm,
                size=size,
                errorbar=errorbar,
                estimator=estimator,
                palette=palette,
                ax=ax,
                **kwargs
            )
        else:
            g = sns.scatterplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                style=style,
                hue_norm=hue_norm,
                size=size,
                errorbar=errorbar,
                estimator=estimator,
                palette=palette,
                ax=ax,
                **kwargs
            )
    else:
        # Create relplot
        g = sns.relplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            col=col,
            row=row,
            style=style,
            hue_norm=hue_norm,
            size=size,
            kind=kind,
            errorbar=errorbar,
            estimator=estimator,
            palette=palette,
            **kwargs
        )
    # Apply log scales unless overridden
    if logx:
        g.set(xscale='log')
    if logy:
        g.set(yscale='log')

    if kind == "line":
        group_vars = [var for var in [hue, col, row, style, size] if var is not None] + [x]
    else:
        group_vars = [var for var in [hue, col, row, style, size] if var is not None]
    if group_vars:
        counts = data.groupby(group_vars).size().value_counts()
        print("Number of samples per group:")
        print(counts)
    else:
        print(f"Total samples: {len(data)} (no grouping applied)")

        return g

def df_vals(df: pd.DataFrame):
    for column in df.columns:
        # if len(column) < 20:
        print(f"'{column}':")
        try:
            # For numeric (float/int) columns, print compact statistical summary
            if pd.api.types.is_categorical_dtype(df[column]):
                print("Categorical: ", df[column].unique())
            elif np.issubdtype(df[column].dtype, np.number) and len(df[column].unique()) > 10:
                stats = df[column].describe()
                print(
                    f"min: {stats['min']:.2f}, max: {stats['max']:.2f}, mean: {stats['mean']:.2f}, std: {stats['std']:.2f}")
            elif np.issubdtype(df[column].dtype, np.number) or df[column].dtype == bool:
                print(df[column].unique())
            elif df[column].dtype == 'object':
                unique_vals = df[column].unique()
                if any(isinstance(val, str) for val in unique_vals):
                    for val in unique_vals:
                        print(f"- {val}")
                else:
                    print(unique_vals)
            # For complex types (arrays, etc.), just print the dtype
            else:
                print(f"Column type: {df[column].dtype}")
        except Exception as e:
            print(e)
            print("The following column failed")
            print(column)
        print("-" * 40)  # separator between columns
    print("Attributes")
    for key, val in df.attrs.items():
        print(key, ": ", val)

def plot_single_image(image, denormalize=True):
    image = image.cpu()
    
    # Denormalize to [0, 1] if requested
    # if denormalize:
    #     image = (image + 1.0) / 2.0

    image = torch.clamp(image, 0, 1)
    np_img = image.permute(1, 2, 0).numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(np_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_vec_mag(vec):
    plt.figure()
    plt.hist(np.abs(vec), bins=np.logspace(np.log10(np.abs(vec[vec != 0]).min()),
                                           np.log10(np.abs(vec).max()), 50))
    plt.grid(True)
    plt.show(block=False)
