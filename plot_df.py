from matplotlib.colors import LogNorm
import seaborn as sns
import numpy as np
import pandas as pd
from utils import repeat
import matplotlib.pyplot as plt

# df = pd.DataFrame({'a': [1,2,3]})
# pd.to_pickle(df, "./data/test.pickle")

df = pd.read_pickle("./data/sweep_fixed.pickle")
assert isinstance(df, pd.DataFrame)

df.columns

df['m'].unique()
df['sqt_n'][0]**2

np.max(df["runtime_seconds"])/60
# 24 seconds

df['rel_m'] = df['m']/df['sqt_n']**2

df['rre']

df['runtime_seconds'].mean()*6/60 # estimate of total runtime for 6 rows per core.

plt.figure()
sns.lineplot(df, x="m", y="rre", hue="sigma", hue_norm=LogNorm(vmin=1e-4, clip=True), palette="plasma", errorbar="se")
plt.xscale("log")
plt.yscale("log")

img=df['recovered_img'][0]

plt.figure()
plt.imshow(df.loc[5]['recovered_img'])
