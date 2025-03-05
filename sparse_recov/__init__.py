from .uneven_sparse_experiment import make_df, experiment
from .data_analysis import plot_on_facets, perform_least_squares_fit, add_fit_lines_to_plot
from .sampling_recov_functions import (
    S_transpose, SF_transpose_map, SF2_transpose_map, SDF2_transpose_map,
    wr_prob_vec, Bernoulli_prob_weights, SF_func, SF2_func, SDF2_func, soft_threshold
)
