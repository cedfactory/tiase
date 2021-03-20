from lib.fimport import *
from lib.findicators import *
from lib.visu import *

df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/_AI.PA.csv")
#visu.DisplayFromDataframe(df,"Close")
print(df.head())
df = findicators.add_technical_indicators(df)
print(list(df.columns))
print(df.head())
