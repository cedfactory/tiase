import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def missing_values(df):
    # Drop the NaNs
    df.dropna(axis=0, how='any', inplace=True)
    # df = df.reset_index(drop=True)
    return df


def drop_duplicates(df):
    df.drop_duplicates(inplace=True)
    # df = df.reset_index(drop=True)
    return df


def normalize_outliers_std_cutoff(df, n_sigmas):
    d1 = pd.DataFrame(df['close'])
    d1['simple_rtn'] = d1.close.pct_change()
    d1_mean = d1['simple_rtn'].agg(['mean', 'std'])

    mu = d1_mean.loc['mean']
    sigma = d1_mean.loc['std']

    cond = (d1['simple_rtn'] > mu + sigma * n_sigmas) | (d1['simple_rtn'] < mu - sigma * n_sigmas)
    d1['outliers'] = np.where(cond, 1, 0)

    nb_outliers = d1.outliers.value_counts()
    print(nb_outliers)

    d1['sign_simple_rtn'] = np.where(d1['simple_rtn'] > 0, 1, -1)
    d1['new_simple_rtn'] = np.where(d1['outliers'] == 1, mu + sigma * n_sigmas * d1['sign_simple_rtn'],
                                    d1['simple_rtn'])

    d1['close_shift'] = d1['close'].shift(1)
    d1['new_rtn'] = (d1['close'] - d1['close_shift']) / d1['close_shift']
    d1['new_close'] = d1['close_shift'] + d1['new_simple_rtn'] * d1['close_shift']
    d1['test'] = d1['new_close'] - d1['close']
    d1['new_close'][0] = d1['close'][0]

    df['close'] = d1['new_close'].copy()

    return df


def normalize_outliers_winsorize(df, outlier_cutoff):
    d1 = pd.DataFrame(df['close'])
    d1['simple_rtn'] = d1.close.pct_change()
    d1_mean = d1['simple_rtn'].agg(['mean', 'std'])

    mu = d1_mean.loc['mean']
    sigma = d1_mean.loc['std']
    n_sigmas = 3

    cond = (d1['simple_rtn'] > mu + sigma * n_sigmas) | (d1['simple_rtn'] < mu - sigma * n_sigmas)
    d1['outlier'] = np.where(cond, 1, 0)

    nb_outliers = d1.outlier.value_counts()
    print(nb_outliers)

    d2 = pd.DataFrame(d1['simple_rtn'])

    d2.pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                             upper=x.quantile(1 - outlier_cutoff),
                             axis=1,
                             inplace=True))

    d1['close_shift'] = d1['close'].shift(1)
    d1['new_rtn'] = d2['simple_rtn'].copy()
    # d1['new_rtn'] = (d1['close'] - d1['close_shift']) / d1['close_shift']
    d1['new_close'] = d1['close_shift'] + d1['new_rtn'] * d1['close_shift']
    d1['test'] = d1['new_close'] - d1['close']
    d1['new_close'][0] = d1['close'][0]
    df['close'] = d1['new_close'].copy()

    return df


def normalize_outliers_mam(df, n_sigmas):
    # Using Moving Average Mean and Standard Deviation as the Boundary
    d1 = pd.DataFrame(df['close'])
    d1['simple_rtn'] = d1.close.pct_change()
    d1_mean = d1['simple_rtn'].agg(['mean', 'std'])

    d1[['mean', 'std']] = d1['simple_rtn'].rolling(window=21).agg(['mean', 'std'])
    # d1.dropna(inplace=True)

    cond = (d1['simple_rtn'] > d1['mean'] + d1['std'] * n_sigmas) \
           | (d1['simple_rtn'] < d1['mean'] - d1['std'] * n_sigmas)
    d1['outliers'] = np.where(cond, 1, 0)

    nb_outliers = d1.outliers.value_counts()
    print(nb_outliers)

    d1['sign_simple_rtn'] = np.where(d1['simple_rtn'] > 0, 1, -1)
    d1['new_simple_rtn'] = np.where(d1['outliers'] == 1, d1['mean'] + d1['std'] * n_sigmas * d1['sign_simple_rtn'],
                                    d1['simple_rtn'])

    d1['close_shift'] = d1['close'].shift(1)
    d1['new_rtn'] = (d1['close'] - d1['close_shift']) / d1['close_shift']
    d1['new_close'] = d1['close_shift'] + d1['new_simple_rtn'] * d1['close_shift']
    d1['test'] = d1['new_close'] - d1['close']
    d1['new_close'][0] = d1['close'][0]

    df['close'] = d1['new_close'].copy()

    return df


def normalize_outliers_ema(df, n_sigmas):
    # Using EMA and Standard Deviation as the Boundary
    d1 = pd.DataFrame(df['close'])
    d1['simple_rtn'] = d1.close.pct_change()
    d1[['mean', 'std']] = d1['simple_rtn'].ewm(span=21).agg(['mean', 'std'])
    # d1.dropna(inplace=True)

    condition = (d1['simple_rtn'] > d1['mean'] + d1['std'] * n_sigmas) | (
                d1['simple_rtn'] < d1['mean'] - d1['std'] * n_sigmas)
    d1['outliers'] = np.where(condition, 1, 0)

    nb_outliers = d1.outliers.value_counts()
    print(nb_outliers)

    d1['sign_simple_rtn'] = np.where(d1['simple_rtn'] > 0, 1, -1)
    d1['new_simple_rtn'] = np.where(d1['outliers'] == 1, d1['mean'] + d1['std'] * n_sigmas * d1['sign_simple_rtn'],
                                    d1['simple_rtn'])

    d1['close_shift'] = d1['close'].shift(1)
    d1['new_rtn'] = (d1['close'] - d1['close_shift']) / d1['close_shift']
    d1['new_close'] = d1['close_shift'] + d1['new_simple_rtn'] * d1['close_shift']
    d1['test'] = d1['new_close'] - d1['close']
    d1['new_close'][0] = d1['close'][0]
    df['close'] = d1['new_close'].copy()

    return df


def feature_encoding(df):
    """
    Not implemented
    """
    # creating instance of one-hot-encoder
    # enc = OneHotEncoder(handle_unknown='ignore')

    # passing bridge-types-cat column (label encoded values of bridge_types)
    # enc_df = pd.DataFrame(enc.fit_transform(df[['Bridge_Types_Cat']]).toarray())

    # merge with main df bridge_df on key values
    # df = enc_df.join(enc_df)

    return df


def data_log_transformation(df, columns):
    # Do the logarithm trasnformations for required features

    for col in columns:
        plot_histogram(df, col, 'histogram_log_1_' + str(col) + '.png')

    logarithm_transformer = FunctionTransformer(np.log1p, validate=True)

    to_right_skewed = logarithm_transformer.transform(df[columns])

    i = 0
    for column in columns:
        df[column] = to_right_skewed[:, i]
        i = i + 1

    for col in columns:
        plot_histogram(df, col, 'histogram_log_2_' + str(col) + '.png')

    return df


def data_x2_transformation(df, columns):
    for col in columns:
        plot_histogram(df, col, 'histogram_xx_1_' + str(col) + '.png')

    # Do the x2 trasnformations for required features
    exp_transformer = FunctionTransformer(lambda x: x ** 2, validate=True)

    # apply the transformation to your data
    to_left_skewed = exp_transformer.transform(df[columns])

    i = 0
    for column in columns:
        df[column] = to_left_skewed[:, i]
        i = i + 1

    for col in columns:
        plot_histogram(df, col, 'histogram_xx_2_' + str(col) + '.png')

    return df


def data_scaling(df):
    return df
