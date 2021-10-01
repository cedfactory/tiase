import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from tiar.fimport import visu
from rich import print,inspect

def missing_values(df):
    df['inf'] = 0
    for col in df.columns:
        df['inf'] = np.where((df[col] == np.inf) | (df[col] == -np.inf), 1, df['inf'])

    df = df.drop(df[df.inf == 1].index)
    df = df.drop(['inf'], axis=1)

    df.replace([np.inf, -np.inf], np.nan)
    # Drop the NaNs
    df.dropna(axis=0, how='any', inplace=True)

    return df


def drop_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

# reference : https://python.plainenglish.io/identifying-outliers-part-one-c0a31d9faefa
def normalize_outliers_std_cutoff(df, n_sigmas):
    d1 = pd.DataFrame(df['close'].copy())
    d1['simple_rtn'] = d1.close.pct_change()
    d1_mean = d1['simple_rtn'].agg(['mean', 'std'])

    mu = d1_mean['mean']
    sigma = d1_mean['std']

    # normalize simple_rtn
    d1["simple_rtn_normalized"] = d1["simple_rtn"].clip(lower = mu - sigma * n_sigmas, upper = mu + sigma * n_sigmas)
    # revert pct_change to compute new close values
    d1['close'] = df['close'][0] * (1. + d1['simple_rtn_normalized']).cumprod()
    d1['close'][0] = df['close'][0]

    # update close values in df
    df['close'] = d1["close"]

    return df


def cut_outliers_std_cutoff(df, n_sigmas):
    d1 = pd.DataFrame(df['close'].copy())
    d1['simple_rtn'] = d1.close.pct_change()
    d1_mean = d1['simple_rtn'].agg(['mean', 'std'])

    mu = d1_mean.loc['mean']
    sigma = d1_mean.loc['std']

    cond = (d1['simple_rtn'] > mu + sigma * n_sigmas) | (d1['simple_rtn'] < mu - sigma * n_sigmas)
    d1['outliers'] = np.where(cond, 1, 0)

    nb_outliers = d1.outliers.value_counts()
    print(nb_outliers)

    # drop all outliers raws
    df.drop(df[d1['outliers'] == 1].index, inplace=True)

    return df

# reference : https://python.plainenglish.io/identifying-outliers-part-one-c0a31d9faefa
def normalize_outliers_winsorize(df, outlier_cutoff):
    d1 = pd.DataFrame(df['close'].copy())
    d1['simple_rtn'] = d1.close.pct_change()

    d2 = pd.DataFrame(d1['simple_rtn'].copy())

    # normalize simple_rtn
    d2.pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                             upper=x.quantile(1 - outlier_cutoff),
                             axis=1, inplace=True))

    # revert pct_change to compute new close values
    d2['close'] = df['close'][0] * (1. + d2['simple_rtn']).cumprod()
    d2['close'][0] = df['close'][0]

    # update close values in df
    df['close'] = d2["close"]

    return df

def normalize_outliers_mam(df, n_sigmas):
    # Using Moving Average Mean and Standard Deviation as the Boundary
    d1 = pd.DataFrame(df['close'].copy())
    d1['simple_rtn'] = d1.close.pct_change()

    d1[['mean', 'std']] = d1['simple_rtn'].rolling(window=21).agg(['mean', 'std'])

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
    d1['new_close'][0] = d1['close'][0]

    df['close'] = d1['new_close'].copy()
    df['simple_rtn'] = d1['new_rtn'].copy()

    return df


def normalize_outliers_ema(df, n_sigmas):
    # Using EMA and Standard Deviation as the Boundary
    d1 = pd.DataFrame(df['close'].copy())
    d1['simple_rtn'] = d1.close.pct_change()
    d1[['mean', 'std']] = d1['simple_rtn'].ewm(span=21).agg(['mean', 'std'])

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
    d1['new_close'][0] = d1['close'][0]
    df['close'] = d1['new_close'].copy()
    df['simple_rtn'] = d1['new_rtn'].copy()

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
    logarithm_transformer = FunctionTransformer(np.log1p, validate=True)
    to_right_skewed = logarithm_transformer.transform(df[columns])

    i = 0
    for column in columns:
        df[column] = to_right_skewed[:, i]
        i = i + 1

    return df


def data_x2_transformation(df, columns):
    # Do the x2 trasnformations for required features
    exp_transformer = FunctionTransformer(lambda x: x ** 2, validate=True)
    to_left_skewed = exp_transformer.transform(df[columns])

    i = 0
    for column in columns:
        df[column] = to_left_skewed[:, i]
        i = i + 1

    return df


def data_scaling(df):
    return df


def drop_ohlcv(df):
    df = df.drop(['open', 'close', 'high', 'low', 'adj_close', 'volume'], axis=1)
    return df
