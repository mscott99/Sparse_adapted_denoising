from scipy import sparse
from scipy.fft import fft, fft2, ifft, ifft2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from seaborn.axisgrid import FacetGrid
from scipy import stats
import seaborn as sns
import pandas as pd


def perform_least_squares_fit(df, x_col, y_col, group_cols, min_x=None):
    """
    Perform least squares fit on grouped data, including x range information

    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    x_col : str
        Name of x column
    y_col : str
        Name of y column
    group_cols : list
        List of column names to group by
    min_x : float, optional
        Minimum value for x to include in fit

    Returns:
    --------
    pandas DataFrame
        DataFrame containing fit results and x range information
    """
    # Filter data if min_x is specified
    df_filtered = df[df[x_col] > min_x].copy(
    ) if min_x is not None else df.copy()

    fit_results = []

    # Group by the specified parameters
    groups = df_filtered.groupby(group_cols)

    # Perform least squares fit for each group
    for name, group in groups:
        # Convert group parameters to dictionary
        params = dict(
            zip(group_cols, name if isinstance(name, tuple) else [name]))
        # Get x range information
        x_min = group[x_col].min()
        x_max = group[x_col].max()

        # Perform linear fit on log-transformed data
        x = np.log10(group[x_col])
        y = np.log10(group[y_col])

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        fit_results.append({
            **params,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'x_min': x_min,
            'x_max': x_max
        })

    return pd.DataFrame(fit_results)


def add_fit_lines_to_plot(subplot_fits, ax,
                          line_style='--',
                          alpha=0.5,
                          color='red',
                          annotate_slopes=True,
                          slope_precision=2,
                          stats_precision=2):
    # Plotting remains the same
    for _, fit in subplot_fits.iterrows():
        x_range = np.logspace(
            np.log10(fit['x_min']), np.log10(fit['x_max']), 100)
        y_fit = 10**(fit['intercept'] + fit['slope'] * np.log10(x_range))
        ax.plot(x_range, y_fit, line_style, alpha=alpha, color=color)

    # Annotation handling
    if annotate_slopes and not subplot_fits.empty:
        slopes = subplot_fits['slope']
        n_slopes = len(slopes)

        # Build annotation text with null checks
        text_parts = []
        if n_slopes > 0:
            if n_slopes > 1:
                slope_list = ', '.join(
                    [f'{s:.{slope_precision}f}' for s in slopes])
                text_parts.append(f'Slopes: {slope_list}')
                stats_text = (f'Mean: {np.mean(slopes):.{stats_precision}f} ± '
                              f'{np.std(slopes):.{stats_precision}f}')
                text_parts.append(stats_text)
            else:
                text_parts.append(f'Slope: {slopes[0]:.{slope_precision}f}')

            full_text = '\n'.join(text_parts)

            # Add annotation
            ax.text(
                x=0.95, y=0.05,
                s=full_text,
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=8,
                bbox=dict(
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='lightgray',
                    boxstyle='round,pad=0.2'
                )
            )


def plot_on_facets(
    plt_fn,
    g: sns.FacetGrid,
    fit_df: pd.DataFrame,
    row_label=None,  # Made optional
    col_label=None,  # Made optional
    **kwargs
):
    """
    plot to each facet of a facetgrid by using the relevant data from a DataFrame. 

    Parameters:
    -----------
    row_label : str, optional
        Name of row facet column (if exists)
    col_label : str, optional
        Name of column facet column (if exists)
    """
    if g.row_names == [] and g.col_names == []:
        plt_fn(fit_df, g.ax, **kwargs)
    elif g.row_names != [] and g.col_names == []:
        if row_label == None:
            raise ValueError("The row label is required.")
        for row_val, ax in g.axes_dict.items():
            plt_fn(fit_df[fit_df[row_label] == row_val], ax, **kwargs)
    elif g.col_names != [] and g.row_names == []:
        if col_label == None:
            raise ValueError("The column label is required.")
        for col_val, ax in g.axes_dict.items():
            plt_fn(fit_df[fit_df[col_label] == col_val], ax, **kwargs)
    elif g.row_names != [] and g.col_names != []:
        if row_label == None or col_label == None:
            raise ValueError("Both the row and column labels are required.")
        for (row_val, col_val), ax in g.axes_dict.items():
            plt_fn(fit_df[(fit_df[row_label] == row_val)&(fit_df[col_label] == col_val)], ax, **kwargs)

# Making minimal changes to the above function, incorporate the fact that there may
# not be rows or columns inside the FacetGrid g.

def S_transpose(indices, m, m_tld, n):
    return np.sqrt(n/m) * sparse.coo_matrix(
        (np.ones(indices.shape[0]), (indices, range(m_tld))), shape=(n, m_tld))


def SF_transpose_map(indices, m, n):
    m_tld = indices.size
    S_conj = S_transpose(indices, m, m_tld, n)
    return lambda x: ifft((S_conj @ x.flat), norm="ortho")


def SF2_transpose_map(indices, m, n, shape):
    m_tld = indices.size
    S_conj = S_transpose(indices, m, m_tld, n)
    return lambda x: ifft2((S_conj @ x.flat).reshape(shape), norm="ortho")


def SDF2_transpose_map(indices, m, n, shape, d):
    m_tld = indices.size
    S_conj = S_transpose(indices, m, m_tld, n)
    return lambda x: ifft2((d*(S_conj @ x.flat)).reshape(shape), norm="ortho")


def wr_prob_vec(loc_co):
    return loc_co**2/np.sum(loc_co**2)


def Bernoulli_prob_weights(loc_co: NDArray, m: int) -> Tuple[NDArray, int, float]:
    pos_co = loc_co[loc_co > 0]
    n = pos_co.size
    assert m < n, "Cannot sample enough measurements given entries with zero coherences cannot be sampled"
    sorted_co = np.sort(pos_co.flat)
    J = n

    def Lsqrd(j, m, n, sorted_co):
        return m * np.sum(sorted_co[:j]**2) / (m - (n-j))
    for j in range(n, n-m-1, -1):
        if m*sorted_co[j-1]**2 < Lsqrd(j, m, n, sorted_co):
            J = j
            break
    Lsqr = Lsqrd(J, m, n, sorted_co)
    return np.clip(m * loc_co**2 / Lsqr, 0, 1), J, Lsqr


def SF_func(indices: NDArray, m, n):
    return lambda x: np.sqrt(n/m) * fft(x, norm="ortho")[indices]


def SF2_func(indices: NDArray, m, n):
    return lambda x: np.sqrt(n/m) * fft2(x, norm="ortho").flat[indices]


def SDF2_func(indices, m, n, d):
    return lambda x: np.sqrt(n/m) * (d*fft2(x, norm="ortho").flat)[indices]


def soft_threshold(x, lambda_):
    """Soft thresholding operator"""
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
