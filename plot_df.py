import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.DataFrame({'a': [1,2,3]})
# pd.to_pickle(df, "./data/test.pickle")

df = pd.read_pickle("./data/unif_recov.pickle")

df.columns

np.mean(df["runtime_seconds"])
# 24 seconds

df['sqt_n']**2-1


df['m']

df['rel_m'] = df['m']/df['sqt_n']**2

sns.lineplot(df, x="rel_m", y="rre", hue="sigma")
plt.xscale("log")
plt.yscale("log")

img=df['recovered_img'][0]

plt.figure()
plt.imshow(df.loc[5]['recovered_img'])
