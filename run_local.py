from utils.utils_data import run_local_parallel_experiment
from sparse_recov import make_df, experiment, ExperimentTypes
from datetime import date

df = run_local_parallel_experiment(make_df(), experiment)

df.to_pickle('./data/last_sparse.pickle')
