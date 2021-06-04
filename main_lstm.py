import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *

#
# basic lstm
# Reference : https://github.com/yuhaolee97/stock-project/blob/main/basicmodel.py
#
def basic():
    filename = "./lib/data/CAC40/AI.PA.csv"
    filename = "./google_stocks_data.csv"
    df = fimport.GetDataFrameFromCsv(filename)

    model = lstm_basic.LSTMBasic(df)
    model.create_model()

    analysis = model.get_analysis()
    print("mape : {:.2f}".format(analysis["mape"]))
    print("rmse : {:.2f}".format(analysis["rmse"]))
    print("mse :  {:.2f}".format(analysis["mse"]))

    model.export_predictions("lstm_basic.png")

    model.save_model("basic")

    model_loaded = lstm_basic.LSTMBasic(df, name="basic")
    prediction = model_loaded.predict()
    print(prediction)

def classification():
    print("classification")

    filename = "./google_stocks_data.csv"
    df = fimport.GetDataFrameFromCsv(filename)

    model = lstm_classification.LSTMClassification(df)
    model.create_model()


_usage_str = """
Options:
    [ --basic, --classification]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--basic": basic()
        elif sys.argv[1] == "--classification": classification()
        else: _usage()
    else: _usage()
