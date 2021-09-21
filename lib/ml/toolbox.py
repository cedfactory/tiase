import pandas as pd
import numpy as np
import glob
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
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
    if train_split <= 1:
        split_index = int(df.shape[0] * train_split)
    else:
        split_index = train_split

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

def get_train_test_data_list_from_CV_WF_split_dataframe(df, nb_split=5):
    #tscv = TimeSeriesSplit(gap=0, max_train_size=int(len(df)/2), n_splits=nb_split, test_size=100)
    tscv = TimeSeriesSplit(gap=0, max_train_size=500, n_splits=nb_split, test_size=100)

    list_df_training = []
    list_df_testing = []
    for split_index in tscv.split(df):
        debug = False
        #debug = True
        if(debug):
            print("df size: ", len(df))
            print("TRAIN:", split_index[0][0]," -> ", split_index[0][len(split_index[0])-1], " Size: ", split_index[0][len(split_index[0])-1] - split_index[0][0])
            print("TEST: ", split_index[1][0], " -> ", split_index[1][len(split_index[1]) - 1], " Size: ", split_index[1][len(split_index[1])-1] - split_index[1][0])
            print(" ")
        train = []
        for i in range(0,
                       split_index[0][0],
                       1):
            train.append(-1)
        train.extend(split_index[0].tolist().copy())

        for i in range(len(train),
                       len(df),
                       1):
            train.append(-1)

        test = []
        for i in range(0,
                       split_index[1][0],
                       1):
            test.append(-1)
        test.extend(split_index[1].tolist().copy())

        for i in range(len(test),
                       len(df),
                       1):
            test.append(-1)

        df['train'] = train
        df['test']  = test
        df_train = df[df['train'] != -1]
        df_test = df[df['test'] != -1]
        df.drop(columns=['train'], inplace=True)
        df.drop(columns=['test'], inplace=True)
        df_train.drop(columns=['train'], inplace=True)
        df_train.drop(columns=['test'], inplace=True)
        df_test.drop(columns=['test'], inplace=True)
        df_test.drop(columns=['train'], inplace=True)
        list_df_training.append(df_train)
        list_df_testing.append(df_test)

    return list_df_training, list_df_testing

def add_row_to_df(df,ls):
    """
    Given a dataframe and a list, append the list as a new row to the dataframe.

    :param df: <DataFrame> The original dataframe
    :param ls: <list> The new row to be added
    :return: <DataFrame> The dataframe with the newly appended row
    """

    numEl = len(ls)

    newRow = pd.DataFrame(np.array(ls).reshape(1,numEl), columns = list(df.columns))

    df = df.append(newRow, ignore_index=True)

    return df

def merge_csv(extension):
    #all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    all_filenames = [i for i in glob.glob('*{}'.format(extension))]

    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.to_csv("combined_results.csv", index=False, encoding='utf-8-sig')

'''
Optimal threshold
references :
https://www.sciencedirect.com/science/article/abs/pii/S2214579615000611
https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293
https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
'''

def get_classification_threshold(y_test, y_test_prob):
    """
    Given y_test and y_test_prob

    :y_test df: np.array of expected targets
    :y_test_prob ls: np.array of probabilities
    :return: best threshold
    """

    df = pd.DataFrame()
    df['test'] = y_test.tolist()
    df['pred'] = y_test_prob.tolist()

    df = df.sort_values(by='pred', ascending=False)
    pred_list = df['pred'].copy()
    best_accuracy = 0

    for threshold in pred_list:
        y_test_tmp_pred = (y_test_prob > threshold).astype("int32")
        accuracy = metrics.accuracy_score(y_test, y_test_tmp_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold

def proto_classification_threshold(df):
    split_index = len(df) - 21
    df_train = df.iloc[:split_index, :]
    df_test = df.iloc[split_index+1:, :]

    y_train = df_train['y_test'].copy()
    y_train_prob = df_train['y_test_prob'].copy()

    y_test = df_test['y_test'].copy()
    y_test_prob = df_test['y_test_prob'].copy()

    train_pred_list = df_train['y_test_prob'].copy()
    best_accuracy = 0
    for threshold in train_pred_list:
        y_train_tmp_pred = (np.array(y_train_prob) > threshold).astype("int32")
        #toto = np.array(y_train)
        toto = y_train.tolist()
        titi = y_train_tmp_pred.tolist()
        accuracy = metrics.accuracy_score(toto, titi)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    y_test_tmp_pred = (np.array(y_test_prob) > 0.5).astype("int32")
    accuracy_test_05 = metrics.accuracy_score(y_test.tolist(), y_test_tmp_pred)

    y_test_tmp_pred = (np.array(y_test_prob) > best_threshold).astype("int32")
    accuracy_test_best_train_thrs = metrics.accuracy_score(y_test.tolist(), y_test_tmp_pred)

    test_pred_list = df_test['y_test_prob'].copy()
    best_accuracy_test = 0
    for threshold in test_pred_list:
        y_test_tmp_pred = (np.array(y_test_prob) > threshold).astype("int32")
        accuracy = metrics.accuracy_score(y_test.tolist(), y_test_tmp_pred)
        if accuracy > best_accuracy_test:
            best_accuracy_test = accuracy
            best_threshold = threshold

    print("accuracy train:                     ", best_accuracy)
    print("accuracy test threshold 0.5:        ", accuracy_test_05)
    print("accuracy test threshold from train: ", accuracy_test_best_train_thrs)
    print("accuracy test best threshold:       ", best_accuracy_test)

    return best_threshold
