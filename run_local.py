from utils.utils_data import run_local_parallel_experiment, run_local_df_experiment
from sparse_recov import make_df, experiment, ExperimentTypes
from datetime import date

# df = run_local_parallel_experiment(make_df(), experiment)
df = run_local_df_experiment(make_df(), experiment, catch_errors=True)

df.to_pickle('./data/last_local_robust.pickle')
