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

def get_n_classes(obj, target="target"):
    if isinstance(obj, np.ndarray):
        return len(np.unique(obj))
    elif isinstance(obj, pd.DataFrame):
        return obj[target].nunique()
    return 0

def is_multiclass(obj, target="target"):
    return get_n_classes(obj, target) > 2

def add_row_to_df(df,ls):
    """
    Given a dataframe and a list, append the list as a new row to the dataframe.

    :param df: <DataFrame> The original dataframe
    :param ls: <list> The new row to be added
    :return: <DataFrame> The dataframe with the newly appended row
    """

    num_el = len(ls)

    new_row = pd.DataFrame(np.array(ls).reshape(1,num_el), columns = list(df.columns))

    df = df.append(new_row, ignore_index=True)

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

def get_classification_threshold(method, y_test, y_test_prob):
    """
    Given y_test and y_test_prob

    :y_test df: np.array of expected targets
    :y_test_prob ls: np.array of probabilities
    :return: best threshold
    """

    threshold = -1.
    y_test_pred =  []

    if method == "naive":
        threshold = .5

    elif method == "best_accuracy_score":
        df = pd.DataFrame()
        df['test'] = y_test.tolist()
        df['pred'] = y_test_prob.tolist()

        df = df.sort_values(by='pred', ascending=False)
        pred_list = df['pred'].copy()
        best_accuracy = 0

        for threshold in pred_list:
            y_test_tmp_pred = (y_test_prob > threshold[0]).astype("int32")
            accuracy = metrics.accuracy_score(y_test, y_test_tmp_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold[0]

        threshold = best_threshold
    
    if threshold >= 0.:
        y_test_pred = (y_test_prob > threshold).astype("int32")

    return threshold, y_test_pred
