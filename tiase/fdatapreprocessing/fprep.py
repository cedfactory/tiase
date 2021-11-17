import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from ..fimport import visu
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

#
# outliers :
#
# reference : https://python.plainenglish.io/identifying-outliers-part-one-c0a31d9faefa
#
def normalize_outliers_std_cutoff(df, features, n_sigmas):
    for feature in features:
        feature_info = df[feature].agg(['mean', 'std'])
        mu = feature_info['mean']
        sigma = feature_info['std']
        df[feature] = df[feature].clip(lower = mu - sigma * n_sigmas, upper = mu + sigma * n_sigmas)
        
    return df

def cut_outliers_std_cutoff(df, features, n_sigmas):
    for feature in features:
        feature_mean = df[feature].agg(['mean', 'std'])
        mu = feature_mean['mean']
        sigma = feature_mean['std']
        cond = (df[feature] > mu + sigma * n_sigmas) | (df[feature] < mu - sigma * n_sigmas)
        df = df.drop(df[cond].index)
    
    return df

# reference : https://python.plainenglish.io/identifying-outliers-part-one-c0a31d9faefa
def normalize_outliers_winsorize(df, features, outlier_cutoff):
    for feature in features:
        df_tmp = pd.DataFrame(df[feature].copy())
        df_tmp.pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                upper=x.quantile(1 - outlier_cutoff),
                                axis=1, inplace=True))
        df[feature] = df_tmp[feature]

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

# ref : https://medium.com/swlh/identifying-outliers-part-three-257b09f5940b
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
