from lib.fimport import *
from lib.findicators import *
import pandas as pd
import os

def various():
    print("trend :")
    data = {'close':[20, 21, 23, 19, 18, 24, 25, 26, 27]}
    df = pd.DataFrame(data)
    df = findicators.add_technical_indicators(df, ["trend_1d","trend_4d"])
    trend1 = df.loc[:,'trend_1d'].values
    trend4 = df.loc[:,'trend_4d'].values
    print(trend1)
    print(trend4)


    print("")
    df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/AI.PA.csv")
    visu.DisplayFromDataframe(df,"Close","close.png")
    print(df.head())

    df = findicators.add_technical_indicators(df, ["trend_1d", "ema", "bbands", "dx_30", "on_balance_volume"])
    print(list(df.columns))
    print(df.head(20))

    df = findicators.remove_features(df, ['open', 'high', 'low', 'adj_close', 'volume'])
    print(df.head(20))


#
# parse directory cac40
#
def cac40():
    directory = "./lib/data/CAC40/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = fimport.GetDataFrameFromCsv(directory+"/"+filename)
            df = findicators.add_technical_indicators(df, ["trend_1d","ema","macd","rsi_30","cci_30"])
            trend_ratio = findicators.get_ratio_trend(df)
            value = filename[:len(filename)-4]
            print("{} ({}),{:.2f}".format(value, fimport.cac40[value], trend_ratio))
            visu.DisplayFromDataframe(df, 'close', './tmp/'+value+'_close.png')
            visu.DisplayFromDataframe(df, 'ema', './tmp/'+value+'_ema.png')
            visu.DisplayFromDataframe(df, 'macd', './tmp/'+value+'_macd.png')
            visu.DisplayFromDataframe(df, 'rsi_30', './tmp/'+value+'_rsi_30.png')
            visu.DisplayFromDataframe(df, 'cci_30', './tmp/'+value+'_cci_30.png')
            exit(0)



_usage_str = """
Options:
    [--various --cac40]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--various": various()
        elif sys.argv[1] == "--cac40": cac40()
        else: _usage()
    else: _usage()
