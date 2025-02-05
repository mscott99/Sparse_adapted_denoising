from utils import run_local_df_experiment
from src.uneven_sparse_experiment import make_df, experiment, ExperimentTypes

df = run_local_df_experiment(make_df(), experiment, ExperimentTypes)

df.to_pickle('./data/last_local.pickle')
