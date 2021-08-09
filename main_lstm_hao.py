import pandas as pd
from lib.fimport import fimport
from lib.findicators import findicators
from lib.ml import lstm_hao,analysis

#
# basic lstm
# Reference : https://github.com/yuhaolee97/stock-project/blob/main/basicmodel.py
#
def hao_basic():
    filename = "./lib/data/test/google_stocks_data.csv"
    df = fimport.GetDataFrameFromCsv(filename)

    model = lstm_hao.LSTMHaoBasic(df)
    model.create_model()

    analysis = model.get_analysis()
    print("mape : {:.2f}".format(analysis["mape"]))
    print("rmse : {:.2f}".format(analysis["rmse"]))
    print("mse :  {:.2f}".format(analysis["mse"]))

    model.export_predictions("lstm_basic.png")

    model.save_model("basic")

    model_loaded = lstm_hao.LSTMHaoBasic(df, name="basic")
    prediction = model_loaded.predict()
    print(prediction)

import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def hao_trend():
    filename = "./lib/data/test/google_stocks_data.csv"
    df = fimport.GetDataFrameFromCsv(filename)

    model = lstm_hao.LSTMHaoTrend(df)
    model.create_model()

    model_analysis = model.get_analysis()
    print("mape : {:.2f}".format(model_analysis["mape"]))
    print("rmse : {:.2f}".format(model_analysis["rmse"]))
    print("mse :  {:.2f}".format(model_analysis["mse"]))

    print(model_analysis["history"].history.keys())
    analysis.export_history("lstm", model_analysis["history"])

    model.export_predictions("lstm_trend.png")

    prediction = model.predict()
    print(prediction)

_usage_str = """
Options:
    [ --basic, --trend]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--basic": hao_basic()
        elif sys.argv[1] == "--trend": hao_trend()
        else: _usage()
    else: _usage()
