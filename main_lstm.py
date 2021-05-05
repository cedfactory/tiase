import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *



#
# import data
#

df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/_AI.PA.csv")

#data = [[10,1000], [2,500], [3,600], [2,800], [2,500], [0,700], [1,800], [3,500], [5,900], [7,1000]]
#df = pd.DataFrame(data, columns = ['indicator', 'Adj Close'])
#df.index.name = 'Date'


#
# add technical indicators
#

#df = findicators.add_technical_indicators(df, ["macd", "rsi_30", "cci_30", "dx_30"])

print(df.head())

#
# prepare data for validation and testing
#
seq_len = 20
X_train, y_train, X_test, y_test, y_normaliser = toolbox.get_train_test_data(df, seq_len, 0.6)


#
# create a model
#
model = lstm.create_model_lstm(X_train, y_train, seq_len)
