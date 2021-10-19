import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from tiase.fimport import fimport,synthetic,visu
from tiase.findicators import findicators
from tiase.fdatapreprocessing import fdataprep
from tiase.ml import data_splitter,classifier_lstm,classifier_naive,classifier_svc,classifier_xgboost,classifier_decision_tree,meta_classifier,analysis,toolbox
import os
from rich import print,inspect

import warnings
warnings.simplefilter("ignore")

def evaluate_cross_validation(value):
    filename = "./tiase/data/test/google_stocks_data.csv"
    df = fimport.get_dataframe_from_csv(filename)
    print(df.head())

    df = findicators.normalize_column_headings(df)
    df = toolbox.make_target(df, "pct_change", 7)
    df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])
    target = "target"

    model = classifier_lstm.ClassifierLSTM2(df.copy(), target, params={'epochs': 5})
    results = model.evaluate_cross_validation()
    #print("Averaged accuracy : ", results["average_accuracy"])
    print(results)

def evaluate_classifiers(df, value, verbose=False):
    df = findicators.normalize_column_headings(df)
    df = toolbox.make_target(df, "pct_change", 7)
    df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])

    if verbose:
        print(df.head())

    ds = data_splitter.DataSplitter(df, target="target", seq_len=21)
    ds.split(0.7)
        
    target = "target"
    g_classifiers = [
        #{ "name": "DTC", "classifier" : classifier_decision_tree.ClassifierDecisionTree(df.copy(), target=target, data_splitter=ds, params={'max_depth': None})},
        { "name": "DTC3", "classifier" : classifier_decision_tree.ClassifierDecisionTree(df.copy(), target=target, data_splitter=ds, params={'max_depth': 3})},
        #{ "name": "DTC5", "classifier" : classifier_decision_tree.ClassifierDecisionTree(df.copy(), target=target, data_splitter=ds, params={'max_depth': 5})},
        #{ "name": "LSTM1", "classifier" : classifier_lstm.ClassifierLSTM1(df.copy(), target, ds, params={'epochs': 20})},
        #{ "name": "LSTM2", "classifier" : classifier_lstm.ClassifierLSTM2(df.copy(), target, ds, params={'epochs': 20})},
        #{ "name": "LSTM3", "classifier" : classifier_lstm.ClassifierLSTM3(df.copy(), target, ds, params={'epochs': 20})},
        #{ "name": "LSTM Hao 2020", "classifier" : classifier_lstm.ClassifierLSTMHao2020(df.copy(), target, ds, params={'epochs': 40})},
        #{ "name": "BiLSTM", "classifier" : classifier_lstm.ClassifierBiLSTM(df.copy(), target, ds, params={'epochs': 20})},
        { "name": "SVC", "classifier" : classifier_svc.ClassifierSVC(df.copy(), target, ds)},
        { "name": "SVC_poly", "classifier" : classifier_svc.ClassifierSVC(df.copy(), target, ds, params={'kernel': 'poly'})},
        #{ "name": "XGBoost", "classifier" : classifier_xgboost.ClassifierXGBoost(df.copy(), target, ds)},
        #{ "name": "AlwaysAsPrevious", "classifier" : classifier_naive.ClassifierAlwaysAsPrevious(df.copy(), target, ds)},
        #{ "name": "AlwaysSameClass", "classifier" : classifier_naive.ClassifierAlwaysSameClass(df.copy(), target, ds, params={'class_to_return': 1})}
    ]

    test_vs_pred = []
    for g_classifier in g_classifiers:
        name = g_classifier["name"]
        model = g_classifier["classifier"]

        model.fit()
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
            print("Accuracy :  {:.3f}".format(model_analysis["accuracy"]))
            print("Precision : {:.3f}".format(model_analysis["precision"]))
            print("Recall :    {:.3f}".format(model_analysis["recall"]))
            print("f1_score :  {:.3f}".format(model_analysis["f1_score"]))

        analysis.export_confusion_matrix(model_analysis["confusion_matrix"], name+"_classification_confusion_matrix.png")
        analysis.export_roc_curve(model_analysis["y_test"], model_analysis["y_test_prob"], name+"_classification_roc_curve.png")
        if "history" in model_analysis.keys():
            analysis.export_history(name, model_analysis["history"])

        test_vs_pred.append(analysis.testvspred(name, model_analysis["y_test"], model_analysis["y_test_prob"]))

    analysis.export_roc_curves(test_vs_pred, "roc_curves_"+value+".png", value)


    #
    # ensemble
    #
    estimators = meta_classifier.prepare_models_for_meta_classifier_voting(g_classifiers)
    meta_voting = meta_classifier.MetaClassifierVoting(estimators, data_splitter=ds)
    meta_voting.build()
    meta_voting.fit()
    metamodel_analysis = meta_voting.get_analysis()

    if verbose:
        print("meta_model")
        print("Accuracy :  {:.3f}".format(metamodel_analysis["accuracy"]))
        print("Precision : {:.3f}".format(metamodel_analysis["precision"]))
        print("Recall :    {:.3f}".format(metamodel_analysis["recall"]))
        print("f1_score :  {:.3f}".format(metamodel_analysis["f1_score"]))


def experiment(value):

    filename = "./tiase/data/test/google_stocks_data.csv"
    df = fimport.get_dataframe_from_csv(filename)
    print(df.head())

    #y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
    #df = synthetic.create_dataframe(y, 0.8)
    visu.display_from_dataframe(df,"Close", "close.png")

    evaluate_classifiers(df, "experiment", verbose=True)


def cac40():
    directory = "./tiase/data/CAC40/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            value = filename[:len(filename)-4]
            name = fimport.cac40[value]
            print(name)

            df = fimport.get_dataframe_from_csv(directory+"/"+filename)
            evaluate_classifiers(df, value)


_usage_str = """
Options:
    [--experiment --cac40 --cv]
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
        elif sys.argv[1] == "--cv":
            value = ""
            if len(sys.argv) > 2:
                value = sys.argv[2]
            evaluate_cross_validation(value)
        elif sys.argv[1] == "--cac40":
            cac40()
        else:
            _usage()
    else: _usage()