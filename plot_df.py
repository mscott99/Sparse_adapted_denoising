from matplotlib.colors import LogNorm
import seaborn as sns
import numpy as np
import pandas as pd
from utils import repeat, soft_astype_df
import matplotlib.pyplot as plt

from src.uneven_sparse_experiment import ExperimentTypes

# df = pd.DataFrame({'a': [1,2,3]})
# pd.to_pickle(df, "./data/test.pickle")

df= pd.read_pickle("./data/uneven_17.pickle")

assert isinstance(df, pd.DataFrame) 
soft_astype_df(df, ExperimentTypes)

df.dtypes

print(df['ERROR'][0])


df['runtime_seconds'].mean()
# df2 = pd.read_pickle("./data/local_larger_12.pickle")

# df2.columns

if pd.notna(df['ERROR']).any():
    print("There were errors")
    df = df[df['ERROR'].isna()] # Had errors with m too large
    error_df = df[df['ERROR'].notna()]

df.columns

missing_rrep=df[df['rrep'].isna()]
with_rrep=df[df['rrep'].notna()]

(missing_rrep["adapted_sampling"]==True).any()
# Adapted sampling diverges

# wr Diverges
sns.relplot(with_rrep[with_rrep['sampling_type']=="Ber"], kind = "line", x = "m", y = "rrep", col="adapted_sampling", style = "lambda_", hue="sigma", hue_norm=LogNorm(vmin=1e-4, clip=True), palette="plasma", errorbar="se")
plt.xscale("log")
plt.yscale("log")

indices= [1,2,3,4]
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

df['runtime_seconds'].mean()*100/60 # estimate of total runtime for 6 rows per core. 

plt.figure()
sns.lineplot(df, x="m", y="rrep", hue="sigma", style="img", hue_norm=LogNorm(vmin=1e-4, clip=True), palette="plasma", errorbar="se")
plt.xscale("log")
plt.yscale("log")

img=df['recovered_img'][0]

plt.figure()
plt.imshow(df.loc[5]['recovered_img'])
