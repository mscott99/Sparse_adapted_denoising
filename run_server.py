from utils.utils_data import run_local_parallel_experiment, run_local_df_experiment
from sparse_recov import make_df, experiment, ExperimentTypes
import sys

df = run_local_parallel_experiment(make_df(), experiment)
df.to_pickle(sys.stdout.buffer)
sys.stdout.buffer.flush()
