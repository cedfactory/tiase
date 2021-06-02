import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *


# Reference : https://github.com/yuhaolee97/stock-project/blob/main/basicmodel.py

#
# import data
#

filename = "./lib/data/CAC40/AI.PA.csv"
#filename = "./google_stocks_data.csv"
df = fimport.GetDataFrameFromCsv(filename)

model = lstm_basic.LSTMBasic(df)
model.create_model()

model.save_model("basic")

model_loaded = lstm_basic.LSTMBasic(df, name="basic")
model_loaded.predict()
