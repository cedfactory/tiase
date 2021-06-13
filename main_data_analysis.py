from lib.fimport import *
from lib.findicators import *
from lib.ml import *


df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/AI.PA.csv")
techindicators = ["ema","macd","rsi_30","cci_30","dx_30","trend_1d"]
df = findicators.add_technical_indicators(df, techindicators)
df = findicators.remove_features(df, ["open","close","low","high","volume"])
print(df.head())

seq_len = 21
X_train, y_train, X_test, y_test, y_normaliser = toolbox.get_train_test_data_from_dataframe0(df, seq_len, .5)

print(X_train)
print(y_train)

dr.ExportTSNE(X_train, y_train, "tsne.png")

