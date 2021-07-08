import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *

filename = "./lib/data/test/google_stocks_data.csv"
df = fimport.GetDataFrameFromCsv(filename)

y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
df = synthetic.create_dataframe(y, 0.)
#visu.DisplayFromDataframe(df,"Close", "close.png")

model = classifier_lstm.LSTMClassification(df)
model.create_model(epochs = 10)

model_analysis = model.get_analysis()
print("Precision : ", model_analysis["precision"])
print("Recall : ", model_analysis["recall"])
print("f1_score:", model_analysis["f1_score"])


analysis.export_history("lstm_classification", model_analysis["history"])
analysis.export_confusion_matrix(model_analysis["confusion_matrix"], "lstm_classification_confusion_matrix.png")
analysis.export_roc_curve(model_analysis["y_test"], model_analysis["y_pred"], "lstm_classification_roc_curve.png")
