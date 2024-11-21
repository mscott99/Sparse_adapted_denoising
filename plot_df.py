import seaborn as sns
import numpy as np
import pandas as pd
from utils import repeat
import matplotlib.pyplot as plt

# df = pd.DataFrame({'a': [1,2,3]})
# pd.to_pickle(df, "./data/test.pickle")

df = pd.read_pickle("./data/noisy_fixed_2.pickle")

len(df)
len(df[df['m']==9999])

df.columns

df['m'].unique()

df['sqt_n'][0]**2

np.mean(df["runtime_seconds"])
# 24 seconds

df['rel_m'] = df['m']/df['sqt_n']**2

df['m']
df= repeat(df, 3)

len(df)

df['rre']

sns.lineplot(df, x="rel_m", y="rre", hue="sigma", error)

plt.xscale("log")
plt.yscale("log")
plt.savefig("./test.png")

img=df['recovered_img'][0]

plt.figure()
plt.imshow(df.loc[5]['recovered_img'])
