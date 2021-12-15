from sklearn import metrics
from rich import print,inspect
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from . import toolbox

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
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
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

#
# https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428
# https://arxiv.org/pdf/2008.05756.pdf
# https://www.datascienceblog.net/post/machine-learning/performance-measures-multi-class-problems/
#
def classification_analysis(x_test, y_test, y_test_pred, y_test_prob):
    result = {}

    multiclass = toolbox.is_multiclass(y_test)

    result["X_test"] = x_test
    result["y_test"] = y_test
    result["test_size"] = len(x_test)

    result["y_test_pred"] = y_test_pred
    result["y_test_prob"] = y_test_prob

    result["confusion_matrix"] = metrics.confusion_matrix(y_test, y_test_pred)

    result["pred_pos_rate"] = y_test_pred.sum() / len(y_test_pred)
    result["accuracy"] = metrics.accuracy_score(y_test, y_test_pred)

    average = 'binary'
    if multiclass:
        average = 'macro'
    result["precision"] = metrics.precision_score(y_test, y_test_pred, average=average, zero_division=0)
    result["recall"] = metrics.recall_score(y_test, y_test_pred, average=average, zero_division=0)
    result["f1_score"] = metrics.f1_score(y_test, y_test_pred, average=average, zero_division=0)

    n_split = 4
    y_split_len = int(len(y_test)/n_split)
    for i in range(n_split):
        y_test_split = y_test[i*y_split_len:(i+1)*(y_split_len)]
        y_test_pred_split = y_test_pred[i*y_split_len:(i+1)*(y_split_len)]

        result["pred_pos_rate_" + str(i)] = y_test_pred_split.sum() / y_split_len
        result["accuracy_" + str(i)] = metrics.accuracy_score(y_test_split, y_test_pred_split)
        result["precision_" + str(i)] = metrics.precision_score(y_test_split, y_test_pred_split, average=average, zero_division=0)
        result["recall_" + str(i)] = metrics.recall_score(y_test_split, y_test_pred_split, average=average, zero_division=0)
        result["f1_score_" + str(i)] = metrics.f1_score(y_test_split, y_test_pred_split, average=average, zero_division=0)

    return result

def regression_analysis(model, x_test, y_test, y_normaliser = None):
    result = {}

    y_pred = model.predict(x_test)
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
    if toolbox.is_multiclass(y_test):
        print("!!! export_roc_curve : can't deal with multiclass data")
        return


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

def is_testvspreds_multiclass(testvspreds):
    for testvspred in testvspreds:
        if toolbox.is_multiclass(testvspred.yTest):
            return True
    return False

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def export_roc_curves(testvspreds, filename, value):
    if is_testvspreds_multiclass(testvspreds):
        print("!!! export_roc_curves : can't deal with multiclass data")
        return

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
    if 'accuracy' in history.keys():
        # summarize history for accuracy
        fig = plt.figure(1)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        fig.savefig(name+"_accuracy.png")
        plt.close(1)

    if 'loss' in history.keys():
        # summarize history for loss
        fig = plt.figure(1)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
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
    

def export_classifiers_performances(values_classifiers_results, filename):
    f = open(filename, "w")
    f.write("value;classifier id;Accuracy;Precision;Recall;F1 score\n")

    for value in values_classifiers_results:
        classifiers_results = values_classifiers_results[value]
        for classifier_id in classifiers_results:
            result = classifiers_results[classifier_id]
            f.write("{};{};{};{};{};{}\n".format(value, classifier_id, result["accuracy"], result["precision"], result["recall"], result["f1_score"]))

    f.close()