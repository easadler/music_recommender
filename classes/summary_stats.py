def get_moments(df, kurtosis, skew):
    if skew:
        skew = df.skew(axis=1)
        df['skew'] = skew
    if kurtosis:
        kurtosis = df.kurtosis(axis=1)
        df['kurtosis'] = kurtosis
    return df


def SummaryStats(df, axis=1, kurtosis=False, skew=False):
    if axis == 1:
        df_ss = get_moments(df.describe().T, kurtosis, skew)
    elif axis == 0:
        df_ss = get_moments(df.T.describe().T, kurtosis, skew)
    return df_ss




