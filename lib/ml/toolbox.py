import pandas as pd
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def AddTrend(df, columnSource, columnTarget):
    diff = df[columnSource] - df[columnSource].shift(1)
    df[columnTarget] = diff.gt(0).map({False: 0, True: 1})
    return df
    
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
