from tiar.fimport import fimport
from tiar.findicators import findicators
from tiar.ml import toolbox,dr


df = fimport.get_dataframe_from_csv("./tiar/data/CAC40/AI.PA.csv")
techindicators = ["ema_5","macd","rsi_30","cci_30","dx_30","trend_1d"]
df = findicators.add_technical_indicators(df, techindicators)
df = findicators.remove_features(df, ["open","close","low","high","volume"])
print(df.head())

seq_len = 21
X_train, y_train, X_test, y_test, y_normaliser = toolbox.get_train_test_data_from_dataframe1(df, seq_len, "trend_1d", .5)

print(X_train)
print(y_train)

dr.export_tsne(X_train, y_train, "tsne.png")

