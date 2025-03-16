import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List
from seaborn.axisgrid import FacetGrid
from scipy import stats
import seaborn as sns
import pandas as pd


def perform_log_least_squares_fit(df, x_col, y_col, group_cols, min_x=None):
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


def add_log_fit_line_to_plot(subplot_fits, ax,
                          line_style='--',
                          alpha=0.5,
                          color='red',
                          annotate_slopes=True,
                          slope_precision=2,
                          stats_precision=2):
    """
    Add regression lines to a plot based on fit data
    
    Parameters:
    -----------
    subplot_fits : pandas DataFrame
        DataFrame with fit results
    ax : matplotlib Axes
        The axes to plot on
    line_style : str, optional
        Line style for fit lines
    alpha : float, optional
        Transparency for fit lines
    color : str, optional
        Color for fit lines
    annotate_slopes : bool, optional
        Whether to add slope annotations
    slope_precision : int, optional
        Decimal precision for slope values in annotation
    stats_precision : int, optional
        Decimal precision for statistics in annotation
    """
    # Plot fit lines
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

            # Add annotation at top right
            ax.text(
                x=0.95, y=0.95, 
                s=full_text,
                transform=ax.transAxes,
                ha='right',
                va='top',
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
    row_label=None,
    col_label=None,
    **kwargs
):
    """
    Plot to each facet of a facetgrid by using the relevant data from a DataFrame.

    Parameters:
    -----------
    plt_fn : callable
        Function to apply to each facet
    g : sns.FacetGrid
        The facet grid to plot on
    fit_df : pd.DataFrame
        DataFrame containing data to plot
    row_label : str, optional
        Name of row facet column (if exists)
    col_label : str, optional
        Name of column facet column (if exists)
    **kwargs : 
        Additional keyword arguments to pass to plt_fn
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
