import pandas as pd
import numpy as np
from lib.fimport import *
from lib.findicators import *
from lib.ml import *

filename = "./lib/data/test/google_stocks_data.csv"
df = fimport.GetDataFrameFromCsv(filename)

y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
df = synthetic.create_dataframe(y, 0.)
#visu.DisplayFromDataframe(df,"Close", "close.png")

print(df.head())

gClassifiers = [
    { "name": "LSTM", "classifier" : classifier_lstm.ClassifierLSTM(df.copy(), params={'epochs': 20})},
    { "name": "LSTM2", "classifier" : classifier_lstm.ClassifierLSTM2(df.copy())},
    { "name": "XGBoost", "classifier" : classifier_xgboost.ClassifierXGBoost(df.copy())},
    { "name": "SVC", "classifier" : classifier_svc.ClassifierSVC(df.copy())}
]

TestVSPred = []
for gClassifier in gClassifiers:
    name = gClassifier["name"]
    model = gClassifier["classifier"]
    model.create_model()

    model_analysis = model.get_analysis()
    print("Precision : ", model_analysis["precision"])
    print("Recall : ", model_analysis["recall"])
    print("f1_score:", model_analysis["f1_score"])

    analysis.export_confusion_matrix(model_analysis["confusion_matrix"], name+"_lstm_classification_confusion_matrix.png")
    analysis.export_roc_curve(model_analysis["y_test"], model_analysis["y_pred"], name+"_lstm_classification_roc_curve.png")

    TestVSPred.append(analysis.testvspred(name, model_analysis["y_test"], model_analysis["y_pred"]))

analysis.export_roc_curves(TestVSPred, "roc_curves.png")
