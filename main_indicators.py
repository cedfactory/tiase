from lib.fimport import *
from lib.findicators import *

df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/_AI.PA.csv")
visu.ExportFromDataframe(df,"Close","close.png")
print(df.head())

df = findicators.add_technical_indicators(df, ["trend", "ema", "bbands", "macd", "rsi_30", "cci_30", "dx_30"])
print(list(df.columns))
print(df.head(20))

df = findicators.remove_features(df, ['open', 'high', 'low', 'adj close', 'volume'])
print(df.head(20))
