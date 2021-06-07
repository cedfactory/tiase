from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import mean_squared_error

from collections import namedtuple

import numpy as np

import matplotlib.pyplot as plt

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



def classification_analysis(model, X_test, y_test):

    result = {}

    y_pred = model.predict(X_test)
    result["y_pred"] = y_pred

    score = model.score(X_test, y_test)
    result["score"] = score

    f1score = f1_score(y_test, y_pred, average='binary')
    result["f1score"] = f1score

    result["confusion_matrix"] = confusion_matrix(y_test, y_pred)
    #cm_display = ConfusionMatrixDisplay(cm).plot()

    return result

def regression_analysis(model, X_test, y_test, y_normaliser = None):
    result = {}

    y_pred = model.predict(X_test)
    if y_normaliser != None:
        y_pred = y_normaliser.inverse_transform(y_pred)
    result["y_pred"] = y_pred

    result["mape"] = get_mape(y_test, y_pred)
    result["rmse"] = get_rmse(y_test, y_pred)
    result["mse"]  = mean_squared_error(y_test, y_pred)

    return result

#
# ROC curves
#
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
