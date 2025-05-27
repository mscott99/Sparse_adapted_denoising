from utils_plot import log_relplot
from seaborn import relplot
import numpy as np
import scipy
import matplotlib.pyplot as plt

def discrepancy(this_df):
    stats = this_df.groupby("sampling_type").agg({"re":["std", "count", "mean"]})['re']
    se = np.sqrt((stats.loc['adjusted_Ber']['std']**2 + stats.loc['wor']['std']**2)/stats.loc['adjusted_Ber']['count'])
    pos = (stats['mean'].loc['wor'] - stats['mean'].loc['adjusted_Ber'])/se
    return 1 - scipy.special.erf(pos)

def plot_and_label(df, geomean=False, **kwargs):
    df['sampling_type'] = df['sampling_type'].astype("category").cat.rename_categories({"wor":"without replacement", "adjusted_Ber": "conditional Bernoulli", "wr":"with replacement", "Ber":"Bernoulli"})
    df["Sampling type"] = df["sampling_type"]
    if geomean:
        g = log_relplot(df, x="m", y="re", hue = "sampling_type", kind = "line", **kwargs)
    else:
        g = relplot(df, x="m", y="re", hue = "sampling_type", kind = "line", **kwargs)
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("Relative recovery error")
    return g
