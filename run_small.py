from sparse_recov import make_df, experiment
if __name__ == "__main__":
    df = make_df()
    first_row = df.loc[0]
    if 'max_iter' in first_row:
        first_row['max_iter'] = min(10, first_row['max_iter'])
    experiment({**df.attrs, **first_row})  # test for runtime errors
