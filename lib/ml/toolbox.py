import pandas as pd
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing

# from https://github.com/yuhaolee97/stock-project/blob/main/basicmodel.py
def get_train_test_data(df, seq_len, train_split):
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
    next_day_close_values = np.array([train_data['Adj Close'][i + seq_len].copy() for i in range(len(train_data) - seq_len)])
    next_day_close_values = np.expand_dims(next_day_close_values, -1)

    y_normaliser.fit(next_day_close_values)

     
    X_test = np.array([test_normalised_data[:,0:][i  : i + seq_len].copy() for i in range(len(test_normalised_data) - seq_len)])
    

    y_test = np.array([test_data['Adj Close'][i + seq_len].copy() for i in range(len(test_data) - seq_len)])
    
    y_test = np.expand_dims(y_test, -1)

    return X_train, y_train, X_test, y_test, y_normaliser
  
def get_mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE)
    INPUT:
    y_true - actual variable
    y_pred - predicted variable
    OUTPUT:
    mape - Mean Absolute Percentage Error (%)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def get_rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE)
    INPUT:
    y_true - actual variable
    y_pred - predicted variable
    OUTPUT:
    rmse - Root Mean Squared Error
    """
    rmse = np.sqrt(np.mean(np.power((y_true - y_pred),2)))
    return rmse

colorSet = ['orange', 'greenyellow', 'deepskyblue', 'darkviolet', 'crimson', 'darkslategray', 'indigo', 'navy', 'brown',
            'palevioletred', 'mediumseagreen', 'k', 'darkgoldenrod', 'g', 'midnightblue', 'c', 'y', 'r', 'b', 'm', 'lawngreen',
            'mediumturquoise', 'lime', 'teal', 'drive', 'sienna', 'sandybrown']

testvspred = namedtuple('testvspred', ['classifiername', 'yTest', 'yPred'])
def ExportROCCurve(testvspreds, filename):
    idfigroc = 1
    fig = plt.figure(idfigroc)
    plt.title('ROC-curve for {}'.format("test"))
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    cmpt=0
    for testvspred in testvspreds:
        fpr, tpr, thresholds_tree = roc_curve(testvspred.yTest, testvspred.yPred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=colorSet[cmpt], label='{} (auc = {:.2f})'.format(testvspred.classifiername, roc_auc))
        cmpt += 1
    plt.plot([0,1],[0,1], color='lightgrey', label='Luck', linestyle="--")
    plt.legend(loc='lower right', prop={'size':8})
    fig.savefig(filename)
    plt.close(idfigroc)
