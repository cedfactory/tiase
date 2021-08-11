from sklearn import metrics
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



def classification_analysis(X_test, y_test, y_test_pred, y_test_prob):
    result = {}

    result["X_test"] = X_test
    result["y_test"] = y_test
    result["y_test_pred"] = y_test_pred
    result["y_test_prob"] = y_test_prob

    result["confusion_matrix"] = metrics.confusion_matrix(y_test, y_test_pred)

    result["precision"] = metrics.precision_score(y_test, y_test_pred)
    result["recall"] = metrics.recall_score(y_test, y_test_pred)
    result["f1_score"] = metrics.f1_score(y_test, y_test_pred, average="binary")

    return result

def regression_analysis(model, X_test, y_test, y_normaliser = None):
    result = {}

    y_pred = model.predict(X_test)
    if y_normaliser != None:
        y_pred = y_normaliser.inverse_transform(y_pred)
    result["y_pred"] = y_pred

    result["mape"] = get_mape(y_test, y_pred)
    result["rmse"] = get_rmse(y_test, y_pred)
    result["mse"]  = metrics.mean_squared_error(y_test, y_pred)

    return result

#
# ROC curves
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
def export_roc_curve(y_test, y_pred, filename):
    idfigroc = 1
    fig = plt.figure(idfigroc)
    plt.title('ROC-curve for {}'.format("classifier"))
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    fpr, tpr, thresholds_tree = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color='orange', label='{} (auc = {:.2f})'.format("classifier", roc_auc))
    plt.plot([0,1],[0,1], color='lightgrey', label='Luck', linestyle="--")
    plt.legend(loc='lower right', prop={'size':8})
    fig.savefig(filename)
    plt.close(idfigroc)

colorSet = ['orange', 'greenyellow', 'deepskyblue', 'darkviolet', 'crimson', 'darkslategray', 'indigo', 'navy', 'brown',
            'palevioletred', 'mediumseagreen', 'k', 'darkgoldenrod', 'g', 'midnightblue', 'c', 'y', 'r', 'b', 'm', 'lawngreen',
            'mediumturquoise', 'lime', 'teal', 'drive', 'sienna', 'sandybrown']

testvspred = namedtuple('testvspred', ['classifiername', 'yTest', 'yPred'])
def export_roc_curves(testvspreds, filename, value):
    idfigroc = 1
    fig = plt.figure(idfigroc)
    plt.title('ROC-curve for {}'.format(value))
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    cmpt=0
    for testvspred in testvspreds:
        fpr, tpr, thresholds_tree = metrics.roc_curve(testvspred.yTest, testvspred.yPred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=colorSet[cmpt], label='{} (auc = {:.2f})'.format(testvspred.classifiername, roc_auc))
        cmpt += 1
    plt.plot([0,1],[0,1], color='lightgrey', label='Luck', linestyle="--")
    plt.legend(loc='lower right', prop={'size':8})
    fig.savefig(filename)
    plt.close(idfigroc)

def export_history(name, history):
    #print(history.history["history"].keys())

    if ('accuracy' in history.history.keys()):
        # summarize history for accuracy
        fig = plt.figure(1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        fig.savefig(name+"_accuracy.png")
        plt.close(1)

    if ('loss' in history.history.keys()):
        # summarize history for loss
        fig = plt.figure(1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        fig.savefig(name+"_loss.png")
        plt.close(1)

def export_confusion_matrix(confusion_matrix, filename):
    fig = plt.figure(1)
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    #plt.show()
    fig.savefig(filename)
    plt.close(1)
    