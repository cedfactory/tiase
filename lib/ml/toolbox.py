import pandas as pd
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing

#
# from https://github.com/yuhaolee97/stock-project/blob/main/basicmodel.py
#
def get_train_test_data_from_dataframe0(df, seq_len, train_split):
    #Preparation of train test set.
    train_indices = int(df.shape[0] * train_split)

    train_data = df[:train_indices]
    test_data = df[train_indices:]

    test_data = test_data.reset_index()
    test_data = test_data.drop(columns = ['Date'])

    normaliser = preprocessing.MinMaxScaler()
    train_normalised_data = normaliser.fit_transform(train_data)

    test_normalised_data = normaliser.transform(test_data)

    X_train = np.array([train_normalised_data[:,0:][i : i + seq_len].copy() for i in range(len(train_normalised_data) - seq_len)])

    y_train = np.array([train_normalised_data[:,0][i + seq_len].copy() for i in range(len(train_normalised_data) - seq_len)])
    y_train = np.expand_dims(y_train, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    next_day_close_values = np.array([train_data['adj close'][i + seq_len].copy() for i in range(len(train_data) - seq_len)])
    next_day_close_values = np.expand_dims(next_day_close_values, -1)

    y_normaliser.fit(next_day_close_values)

     
    X_test = np.array([test_normalised_data[:,0:][i  : i + seq_len].copy() for i in range(len(test_normalised_data) - seq_len)])
    

    y_test = np.array([test_data['adj close'][i + seq_len].copy() for i in range(len(test_data) - seq_len)])
    
    y_test = np.expand_dims(y_test, -1)

    return X_train, y_train, X_test, y_test, y_normaliser

#
# https://medium.com/analytics-vidhya/analysis-of-stock-price-predictions-using-lstm-models-f993faa524c4
#
def get_train_test_data_from_dataframe(df, seq_len, train_split):
    #Preparation of train test set.
    train_indices = int(df.shape[0] * train_split)

    train_data = df[:train_indices]
    test_data = df[train_indices:]

    test_data = test_data.reset_index()
    test_data = test_data.drop(columns = ['Date'])

    normaliser = preprocessing.MinMaxScaler()
    train_normalised_data = normaliser.fit_transform(train_data)

    test_normalised_data = normaliser.transform(test_data)
    

    X_train = np.array([train_normalised_data[i : i + seq_len].copy() for i in range(len(train_normalised_data) - seq_len)])
    y_train = np.array([train_normalised_data[:,0][i + seq_len].copy() for i in range(len(train_normalised_data) - seq_len)])
    X_test = np.array([test_normalised_data[i : i + seq_len].copy() for i in range(len(test_normalised_data) - seq_len)])
    y_test = np.array([test_data['adj close'][i + seq_len].copy() for i in range(len(test_normalised_data) - seq_len)])

    print("test_normalised_data")
    print(test_normalised_data)
    print("X_train")
    print(X_train)
    print("y_train")
    print(y_train)

    y_train = np.expand_dims(y_train, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    next_day_close_values = np.array([train_data['adj close'][i + seq_len].copy() for i in range(len(train_data) - seq_len)])
    next_day_close_values = np.expand_dims(next_day_close_values, -1)

    y_normaliser.fit(next_day_close_values)

     
    X_test = np.array([test_normalised_data[:,0:][i  : i + seq_len].copy() for i in range(len(test_normalised_data) - seq_len)])
    

    y_test = np.array([test_data['adj close'][i + seq_len].copy() for i in range(len(test_data) - seq_len)])
    
    y_test = np.expand_dims(y_test, -1)

    return X_train, y_train, X_test, y_test, y_normaliser
