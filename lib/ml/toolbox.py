import pandas as pd
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import joblib

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

    X_train = np.array([train_normalised_data[:,0:][i : i + seq_len].copy() for i in range(len(train_normalised_data) - seq_len)])

    y_train = np.array([train_normalised_data[:,0][i + seq_len].copy() for i in range(len(train_normalised_data) - seq_len)])
    y_train = np.expand_dims(y_train, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    next_day_close_values = np.array([train_data[column_target][i + seq_len].copy() for i in range(len(train_data) - seq_len)])
    next_day_close_values = np.expand_dims(next_day_close_values, -1)

    y_normaliser.fit(next_day_close_values)

     
    X_test = np.array([test_normalised_data[:,0:][i  : i + seq_len].copy() for i in range(len(test_normalised_data) - seq_len)])
    

    y_test = np.array([test_data[column_target][i + seq_len].copy() for i in range(len(test_data) - seq_len)])
    
    y_test = np.expand_dims(y_test, -1)

    return X_train, y_train, X_test, y_test, x_normaliser, y_normaliser

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

    X_train = np.array([train_normalised_data[:,0][i : i + seq_len].copy() for i in range(len(train_normalised_data) - seq_len)])
    y_train = np.array([train_data[column_target][i + seq_len].copy() for i in range(len(train_data) - seq_len)])
    y_train = np.expand_dims(y_train, 1)

    X_test = np.array([test_normalised_data[:,0][i  : i + seq_len].copy() for i in range(len(test_normalised_data) - seq_len)])
    y_test = np.array([test_data[column_target][i + seq_len].copy() for i in range(len(test_data) - seq_len)])
    y_test = np.expand_dims(y_test, 1)

    return X_train, y_train, X_test, y_test, x_normaliser
