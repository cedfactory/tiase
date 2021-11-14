import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from tiase.fimport import fimport,synthetic,visu
from tiase.findicators import findicators
from tiase.fdatapreprocessing import fdataprep
from tiase.ml import data_splitter,hyper_parameters_tuning,classifiers_factory,meta_classifier,analysis,toolbox
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

    ds = data_splitter.DataSplitterTrainTestSimple(df, target="target", seq_len=21)
    ds.split(0.7)
    model = classifiers_factory.ClassifiersFactory.get_classifier("lstm1", {'epochs': 5})
    ds = data_splitter.DataSplitterForCrossValidation(df.copy(), nb_splits=5)
    results = model.evaluate_cross_validation(ds, target)
    print("Averaged accuracy : ", results["average_accuracy"])
    print(results)

def evaluate_hyper_parameters_tuning():
    filename = "./tiase/data/test/google_stocks_data.csv"
    df = fimport.get_dataframe_from_csv(filename)
    print(df.head())

    df = findicators.normalize_column_headings(df)
    df = toolbox.make_target(df, "pct_change", 7)
    #df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])
    df = findicators.remove_features(df, ["adj_close","low","high","volume"])
    print(df.head())

    ds = data_splitter.DataSplitterTrainTestSimple(df, target="target", seq_len=3)
    ds.split(0.7)

    classifier = classifiers_factory.ClassifiersFactory.get_classifier("lstm1", {'epochs': 10}, ds)
    classifier.build()
    '''classifier.fit()
    classifier_analysis = classifier.get_analysis()
    print("DTC")
    print("Accuracy :  {:.3f}".format(classifier_analysis["accuracy"]))
    print("Precision : {:.3f}".format(classifier_analysis["precision"]))
    print("Recall :    {:.3f}".format(classifier_analysis["recall"]))
    print("f1_score :  {:.3f}".format(classifier_analysis["f1_score"]))'''

    hpt_grid_search = hyper_parameters_tuning.HPTGridSearch(ds, {"classifier":classifier})
    result = hpt_grid_search.fit()
    print(result)

    model_analysis = hpt_grid_search.get_analysis()
    print("Analysis")
    print("Accuracy :  {:.3f}".format(model_analysis["accuracy"]))
    print("Precision : {:.3f}".format(model_analysis["precision"]))
    print("Recall :    {:.3f}".format(model_analysis["recall"]))
    print("f1_score :  {:.3f}".format(model_analysis["f1_score"]))


def evaluate_classifiers(df, value, verbose=False):
    df = findicators.normalize_column_headings(df)
    df = toolbox.make_target(df, "pct_change", 7)
    df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])

    if verbose:
        print(df.head())

    ds = data_splitter.DataSplitterTrainTestSimple(df, target="target", seq_len=21)
    ds.split(0.7)
        
    target = "target"
    g_classifiers = [
        #{ "name": "DTC", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("decision tree", {'max_depth': None}, ds)},
        { "name": "DTC3", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("decision tree", {'max_depth': 3}, ds)},
        #{ "name": "DTC5", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("decision tree", {'max_depth': 5}, ds)},
        { "name": "Gaussian process", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("gaussian process", None, ds)},
        { "name": "Gaussian naive bayes", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("gaussian naive bayes", None, ds)},
        { "name": "mlp", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("mlp", {'hidden_layer_sizes': 80}, ds)},
        #{ "name": "LSTM1", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("lstm1", {'epochs': 20}, ds)},
        #{ "name": "LSTM2", "classifier" : cclassifiers_factory.ClassifiersFactory.get_classifier("lstm2", {'epochs': 20}, ds)},
        #{ "name": "LSTM3", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("lstm3", {'epochs': 20}, ds)},
        #{ "name": "LSTM Hao 2020", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("lstmhao2020", {'epochs': 40}, ds)},
        #{ "name": "BiLSTM", "classifier" : lassifiers_factory.ClassifiersFactory.get_classifier("bilstm", {'epochs': 20}, ds)},
        { "name": "SVC", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("svc", None, ds)},
        { "name": "SVC_poly", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("svc", {'kernel': 'poly'}, ds)},
        #{ "name": "XGBoost", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("xgboost, None, ds)},
        #{ "name": "AlwaysAsPrevious", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("as previous", None, ds)},
        #{ "name": "AlwaysSameClass", "classifier" : classifiers_factory.ClassifiersFactory.get_classifier("same class", {'class_to_return': 1}, ds)}
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

def experiment(value):

    filename = "./tiase/data/test/google_stocks_data.csv"
    df = fimport.get_dataframe_from_csv(filename)
    print(df.head())

    #y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
    #df = synthetic.create_dataframe(y, 0.8)
    visu.display_from_dataframe(df, "Close", "close.png")

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
    [--experiment --cac40 --cv --hpt]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--hpt":
            evaluate_hyper_parameters_tuning()
        elif sys.argv[1] == "--experiment":
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