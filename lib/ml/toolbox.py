import pandas as pd
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import joblib
from ..fdatapreprocessing import fdataprep

def make_target(df, method, n_days):
    diff = df["close"] - df["close"].shift(n_days)
    df["target"] = diff.gt(0).map({False: 0, True: 1})
    df["target"] = df["target"].shift(-n_days)
    df = fdataprep.process_technical_indicators(df, ["missing_values"])
    return df

# filename should have one of the following extension : ['.z', '.gz', '.bz2', '.xz', '.lzma']
def serialize(scaler, filename):
    joblib.dump(scaler, filename)

def deserialize(filename):
    return joblib.load(filename)

#
# from https://github.com/yuhaolee97/stock-project/blob/main/basicmodel.py
#
def get_train_test_data_from_dataframe0(df, seq_len, column_target, train_split):
    #Preparation of train test set.

    train_indices = int(df.shape[0] * train_split)

    train_data = df[:train_indices]
    #train_data = train_data.reset_index()
    #train_data = train_data.drop(columns = ['Date'])

    test_data = df[train_indices:]
    test_data = test_data.reset_index()
    test_data = test_data.drop(columns = ['Date'])

    x_normaliser = preprocessing.MinMaxScaler()

    train_normalised_data = x_normaliser.fit_transform(train_data)
    test_normalised_data = x_normaliser.transform(test_data)

    x_train = np.array([train_normalised_data[:,0:][i : i + seq_len].copy() for i in range(len(train_normalised_data) - seq_len)])

    y_train = np.array([train_normalised_data[:,0][i + seq_len].copy() for i in range(len(train_normalised_data) - seq_len)])
    y_train = np.expand_dims(y_train, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    next_day_close_values = np.array([train_data[column_target][i + seq_len].copy() for i in range(len(train_data) - seq_len)])
    next_day_close_values = np.expand_dims(next_day_close_values, -1)

    y_normaliser.fit(next_day_close_values)

     
    x_test = np.array([test_normalised_data[:,0:][i  : i + seq_len].copy() for i in range(len(test_normalised_data) - seq_len)])
    

    y_test = np.array([test_data[column_target][i + seq_len].copy() for i in range(len(test_data) - seq_len)])
    
    y_test = np.expand_dims(y_test, -1)

    return x_train, y_train, x_test, y_test, x_normaliser, y_normaliser

def get_train_test_data_from_dataframe1(df, seq_len, column_target, train_split):
    #Preparation of train test set.

    train_indices = int(df.shape[0] * train_split)

    train_data = df[:train_indices]
    #train_data = train_data.reset_index()
    #train_data = train_data.drop(columns = ['Date'])

    test_data = df[train_indices:]
    test_data = test_data.reset_index()
    test_data = test_data.drop(columns = ['Date'])

    x_normaliser = preprocessing.MinMaxScaler()

    train_normalised_data = x_normaliser.fit_transform(train_data)
    test_normalised_data = x_normaliser.transform(test_data)

    x_train = np.array([train_normalised_data[:,0][i : i + seq_len].copy() for i in range(len(train_normalised_data) - seq_len)])
    y_train = np.array([train_data[column_target][i + seq_len].copy() for i in range(len(train_data) - seq_len)])
    y_train = np.expand_dims(y_train, 1)

    x_test = np.array([test_normalised_data[:,0][i  : i + seq_len].copy() for i in range(len(test_normalised_data) - seq_len)])
    y_test = np.array([test_data[column_target][i + seq_len].copy() for i in range(len(test_data) - seq_len)])
    y_test = np.expand_dims(y_test, 1)

    return x_train, y_train, x_test, y_test, x_normaliser


'''
build train & test data from a dataframe
Warnings :
- when the X data is computed wih rows \in [ i ... i+seq_len [, y is computed with i+seq_len (next row)
- normalization is performed
'''
def _get_train_test_data_from_dataframe2(features, target, seq_len):
    n_features = features.shape[1]
    x_train = []
    for i in range(len(features) - seq_len):
        seq = []
        for j in range(n_features):
            seq.extend(features[:,j][i : i + seq_len].flatten())
        x_train.append(seq)
    x_train = np.array(x_train)

    y_train = np.array([target["target"][i + seq_len].copy() for i in range(len(target) - seq_len)])
    y_train = np.expand_dims(y_train, 1)
    
    return x_train, y_train

def get_train_test_data_from_dataframe2(df, seq_len, column_target, train_split, debug=False):

    split_index = int(df.shape[0] * train_split)

    # Preparation of train test set.
    features = df.copy().drop(column_target, axis=1)
    target = pd.DataFrame({'target': df[column_target]})

    train_features = features[:split_index]
    train_target = target[:split_index]
    train_target = train_target.reset_index(drop=True)

    test_features = features[split_index:]
    test_target = target[split_index:]
    test_target = test_target.reset_index(drop=True)

    features_normaliser = preprocessing.MinMaxScaler()

    train_normalised_features = features_normaliser.fit_transform(train_features)
    test_normalised_features = features_normaliser.transform(test_features)

    x_train, y_train = _get_train_test_data_from_dataframe2(train_normalised_features, train_target, seq_len)
    x_test, y_test = _get_train_test_data_from_dataframe2(test_normalised_features, test_target, seq_len)

    if debug:
        df.to_csv("df.csv")

        df_x_train = pd.DataFrame(x_train)
        df_y_train = pd.DataFrame(y_train)
        df_x_test = pd.DataFrame(x_test)
        df_y_test = pd.DataFrame(y_test)

        df_x_train.to_csv("x_train.csv")
        df_y_train.to_csv("y_train.csv")

        df_x_test.to_csv("x_test.csv")
        df_y_test.to_csv("y_test.csv")

    return x_train, y_train, x_test, y_test, features_normaliser

'''
build train & test data from a dataframe
Warning :
- when the X data is computed wih rows \in [ i ... i+seq_len-1 ], y is computed with i+seq_len-1 (same row)
'''
def _get_train_test_data_from_dataframe(features, target, seq_len):
    n_features = features.shape[1]
    x_train = []
    for i in range(len(features) - seq_len + 1):
        seq = []
        for j in range(n_features):
            seq.extend(features[:,j][i : i + seq_len].flatten())
        x_train.append(seq)
    x_train = np.array(x_train)

    y_train = np.array([target["target"][i + seq_len - 1].copy() for i in range(len(target) - seq_len + 1)]).astype(int)
    y_train = np.expand_dims(y_train, 1)
    
    return x_train, y_train

def get_train_test_data_from_dataframe(df, seq_len, column_target, train_split, debug=False):
    split_index = int(df.shape[0] * train_split)

    # separate features and targets
    features = df.copy().drop(column_target, axis=1)
    target = pd.DataFrame({'target': df[column_target]})
    #features = df.drop(column_target, axis=1)
    #target = pd.DataFrame({'target': df[column_target]})

    # split training & testing data
    train_features = features[:split_index]
    train_target = target[:split_index]
    train_target = train_target.reset_index(drop=True)
    #train_target = train_target.drop(columns = ['Date'])

    test_features = features[split_index:]
    test_target = target[split_index:]
    test_target = test_target.reset_index(drop=True)
    #test_target = test_features.drop(columns = ['Date'])
    '''
    train_features.drop(train_features.tail(1).index,inplace=True)
    train_target.drop(train_target.tail(1).index,inplace=True)
    test_features.drop(test_features.tail(1).index,inplace=True)
    test_target.drop(test_target.tail(1).index,inplace=True)
    '''
    features_normaliser = preprocessing.MinMaxScaler()

    train_normalised_features = features_normaliser.fit_transform(train_features)
    test_normalised_features = features_normaliser.transform(test_features)

    x_train, y_train = _get_train_test_data_from_dataframe(train_normalised_features, train_target, seq_len)
    x_test, y_test = _get_train_test_data_from_dataframe(test_normalised_features, test_target, seq_len)

    if debug:
        df_x_train = pd.DataFrame(x_train)
        df_y_train = pd.DataFrame(y_train)
        df_x_test = pd.DataFrame(x_test)
        df_y_test = pd.DataFrame(y_test)

        df_x_train.to_csv("x_train.csv")
        df_y_train.to_csv("y_train.csv")

        df_x_test.to_csv("x_test.csv")
        df_y_test.to_csv("y_test.csv")

    return x_train, y_train, x_test, y_test, features_normaliser
