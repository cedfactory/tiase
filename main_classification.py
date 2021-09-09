import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from lib.fimport import fimport,synthetic,visu
from lib.findicators import findicators
from lib.fdatapreprocessing import fdataprep
from lib.ml import classifier_lstm,classifier_naive,classifier_svc,classifier_xgboost,analysis,toolbox
import os
from rich import print,inspect

import warnings
warnings.simplefilter("ignore")


def evaluate_classifiers(df, value, verbose=False):
    df = findicators.normalize_column_headings(df)
    df = toolbox.make_target(df, "pct_change", 7)
    df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])

    if verbose:
        print(df.head())

    target = "target"
    g_classifiers = [
        { "name": "LSTM1", "classifier" : classifier_lstm.ClassifierLSTM1(df.copy(), target, params={'epochs': 20})},
        { "name": "LSTM2", "classifier" : classifier_lstm.ClassifierLSTM2(df.copy(), target, params={'epochs': 20})},
        { "name": "LSTM3", "classifier" : classifier_lstm.ClassifierLSTM3(df.copy(), target, params={'epochs': 20})},
        { "name": "LSTM Hao 2020", "classifier" : classifier_lstm.ClassifierLSTMHao2020(df.copy(), target, params={'epochs': 40})},
        { "name": "BiLSTM", "classifier" : classifier_lstm.ClassifierBiLSTM(df.copy(), target, params={'epochs': 20})},
        { "name": "SVC", "classifier" : classifier_svc.ClassifierSVC(df.copy(), target, params={'seq_len': 50})},
        { "name": "XGBoost", "classifier" : classifier_xgboost.ClassifierXGBoost(df.copy(), target)},
        { "name": "AlwaysAsPrevious", "classifier" : classifier_naive.ClassifierAlwaysAsPrevious(df.copy(), target)},
        { "name": "AlwaysSameClass", "classifier" : classifier_naive.ClassifierAlwaysSameClass(df.copy(), target, params={'class_to_return': 1})}
    ]

    test_vs_pred = []
    for g_classifier in g_classifiers:
        name = g_classifier["name"]
        model = g_classifier["classifier"]

        model.create_model()
        model_analysis = model.get_analysis()

        debug = False
        if debug:
            history = model.get_history()
            print("history loss: ",history.history["loss"])
            print("history accuracy: ", history.history["accuracy"])
            print("history val_loss: ", history.history["val_accuracy"])
            print("history val_accuracy: ", history.history["val_accuracy"])

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model train vs validation loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper right')
            plt.savefig("test.png")

        if verbose:
            print(name)
            print("Precision : ", model_analysis["precision"])
            print("Recall : ", model_analysis["recall"])
            print("f1_score:", model_analysis["f1_score"])

        analysis.export_confusion_matrix(model_analysis["confusion_matrix"], name+"_classification_confusion_matrix.png")
        analysis.export_roc_curve(model_analysis["y_test"], model_analysis["y_test_prob"], name+"_classification_roc_curve.png")
        if "history" in model_analysis.keys():
            analysis.export_history(name, model_analysis["history"])

        test_vs_pred.append(analysis.testvspred(name, model_analysis["y_test"], model_analysis["y_test_prob"]))

    analysis.export_roc_curves(test_vs_pred, "roc_curves_"+value+".png", value)


def experiment(value):

    filename = "./lib/data/test/google_stocks_data.csv"
    df = fimport.get_dataframe_from_csv(filename)
    print(df.head())

    #y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
    #df = synthetic.create_dataframe(y, 0.8)
    visu.display_from_dataframe(df,"Close", "close.png")

    evaluate_classifiers(df, "experiment", verbose=True)


def cac40():
    directory = "./lib/data/CAC40/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            value = filename[:len(filename)-4]
            name = fimport.cac40[value]
            print(name)

            df = fimport.get_dataframe_from_csv(directory+"/"+filename)
            evaluate_classifiers(df, value)


_usage_str = """
Options:
    [--experiment --cac40]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--experiment":
            value = ""
            if len(sys.argv) > 2:
                value = sys.argv[2]
            experiment(value)
        elif sys.argv[1] == "--cac40":
            cac40()
        else:
            _usage()
    else: _usage()