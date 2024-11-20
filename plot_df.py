import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.DataFrame({'a': [1,2,3]})
# pd.to_pickle(df, "./data/test.pickle")

df = pd.read_pickle("./data/first_noise.pickle")
df.columns

sns.lineplot(df, x="m", y="rre", hue="sigma")
plt.xscale("log")
plt.yscale("log")


img=df['recovered_img'][0]

plt.figure()
plt.imshow(df['recovered_img'][0])
