from lib.fimport import *
from lib.findicators import *
import os

df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/AI.PA.csv")
visu.DisplayFromDataframe(df,"Close","close.png")
print(df.head())

df = findicators.add_technical_indicators(df, ["trend_1", "ema", "bbands", "macd", "rsi_30", "cci_30", "dx_30"])
print(list(df.columns))
print(df.head(20))

df = findicators.remove_features(df, ['open', 'high', 'low', 'adj_close', 'volume'])
print(df.head(20))

#
# parse directory cac40
#
directory = "./lib/data/CAC40/"
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = fimport.GetDataFrameFromCsv(directory+"/"+filename)
        df = findicators.add_technical_indicators(df, ["trend_1d","ema","macd","rsi_30","cci_30"])
        trend_counted = df['trend_1d'].value_counts()
        trend_ratio = 100 * trend_counted[1] / (trend_counted[1]+trend_counted[0])
        value = filename[:len(filename)-4]
        print("{} ({}),{:.2f}".format(value, fimport.cac40[value], trend_ratio))
        visu.DisplayFromDataframe(df, 'close', './tmp/'+value+'_close.png')
        visu.DisplayFromDataframe(df, 'ema', './tmp/'+value+'_ema.png')
        visu.DisplayFromDataframe(df, 'macd', './tmp/'+value+'_macd.png')
        visu.DisplayFromDataframe(df, 'rsi_30', './tmp/'+value+'_rsi_30.png')
        visu.DisplayFromDataframe(df, 'cci_30', './tmp/'+value+'_cci_30.png')
