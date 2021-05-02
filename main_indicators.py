from lib.fimport import *
from lib.findicators import *
#from lib.visu import *

df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/_AI.PA.csv")
#visu.DisplayFromDataframe(df,"Close")
print(df.head())
df = findicators.add_technical_indicators(df, ["ema", "bbands", "macd", "rsi_30", "cci_30", "dx_30"])
print(list(df.columns))
print(df.head(20))
