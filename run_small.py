from sparse_recov import make_df, experiment
if __name__ == "__main__":
    df = make_df()
    experiment({**df.attrs, **df.loc[0]})  # test for runtime errors
