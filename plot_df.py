from src.uneven_sparse_experiment import ExperimentTypes
from src.sampling_recov_functions import add_fit_lines_to_grid, add_fit_lines_to_plot, perform_least_squares_fit, plot_on_facets
from matplotlib.colors import LogNorm
import seaborn as sns
import numpy as np
import pandas as pd
from utils import repeat, soft_astype_df
import matplotlib.pyplot as plt

df = pd.read_pickle("./data/jan_2025_21.pickle")

# Example usage:
# First perform the fits

mydf = df[(df['adapted_sampling']==True)]

fit_df = perform_least_squares_fit(
    mydf[mydf['sigma'] > 1e-3],
    x_col='m',
    y_col='re',
    group_cols=['k', 'sigma', 'alg', 'img', 'sampling_type', 'adapted_sampling'],
    min_x=10**3.2
)


# Create the main plot
g = sns.relplot(
    data=mydf,
    kind="line",
    x="m",
    y="re",
    row="alg",
    col = 'k',
    hue="sigma",
    hue_norm=LogNorm(vmin=1e-4, clip=True),
    palette="plasma",
    errorbar="se"
)

g.set(xscale="log", yscale="log")

plot_on_facets(add_fit_lines_to_plot, g, fit_df, col_label="k", row_label="alg")

plt.show()


df['runtime_seconds'].mean() * 100/60
df['runtime_seconds'].describe()

print(df[df['sampling_type'] == 'wr']
      [df['adapted_sampling'] == True]['ERROR'].unique()[0])

# df2 = pd.read_pickle("./data/local_larger_12.pickle")

# df2.columns

if pd.notna(df['ERROR']).any():
    print("There were errors")
    df = df[df['ERROR'].isna()]  # Had errors with m too large
    error_df = df[df['ERROR'].notna()]

df.columns

df['lambda_'].unique()

sns.relplot(df, kind="line", x="m", y="re", col="lambda_", row="sampling_type", style="adapted_sampling",
            hue="sigma", hue_norm=LogNorm(vmin=1e-4, clip=True), palette="plasma", errorbar="se")
plt.xscale("log")
plt.yscale("log")

sns.relplot(df, kind="line", x="m", y="rrep")

plt.xscale("log")
plt.yscale("log")


indices = [1, 2, 3, 4]
np.max(np.unique(indices, return_counts=True)[1])

np.max([])

missing['m'].unique()

len(missing)

assert isinstance(df, pd.DataFrame)

df.columns

df['m'].unique()
df['sqt_n'][0]**2

np.max(df["runtime_seconds"])/60
# 24 seconds

df['sampling_type'].unique()

df['rel_m'] = df['m']/df['sqt_n']**2

df.columns

df['rre']

# estimate of total runtime for 6 rows per core.
df['runtime_seconds'].mean()*100/60

plt.figure()
sns.lineplot(df, x="m", y="rrep", hue="sigma", style="img", hue_norm=LogNorm(
    vmin=1e-4, clip=True), palette="plasma", errorbar="se")
plt.xscale("log")
plt.yscale("log")

img = df['recovered_img'][0]

plt.figure()
plt.imshow(df.loc[5]['recovered_img'])
