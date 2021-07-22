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

    df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/AI.PA.csv")
    visu.DisplayFromDataframe(df,"Close","close.png")
    print(df.head())

    #technical_indicators = ["trend_1d", "ema", "bbands", "dx_30", "on_balance_volume", "williams_%r", "stoch_%k", "stoch_%d"]
    #technical_indicators = ["williams_%r", "stoch_%k", "stoch_%d", "er", "stc", "sma_5", "sma_10", "sma_15", "sma_20"]
    technical_indicators = ["stc", "sma_5", "sma_10", "sma_15", "sma_20", "ema_10", "ema_20", "ema_50", "atr", "adx", "roc"]
    df = findicators.add_technical_indicators(df, technical_indicators)

    df = findicators.remove_features(df, ['open', 'high', 'low', 'adj_close', 'volume'])
    print(df.head(20))

    idx = pd.Index(pd.date_range("19991231", periods=10), name='Date')
    df = pd.DataFrame([1]*10, columns=["Foobar"], index=idx)
    df = findicators.add_temporal_indicators(df, "Date")
    print(df.head())

def stats(value):
    name = ""
    if value in fimport.cac40:
        name = fimport.cac40[value]
    if value in fimport.nasdaq100:
        name = fimport.nasdaq100[value]
    df = fimport.GetDataFrameFromYahoo(value)
    print("{} ({})".format(value, name))
    print(df.head())

    technical_indicators = findicators.get_all_default_technical_indicators()
    df = findicators.add_technical_indicators(df, technical_indicators)
    trend_ratio, true_positive, true_negative, false_positive, false_negative = findicators.get_trend_info(df)
    print("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(trend_ratio, true_positive, true_negative, false_positive, false_negative))

    visu.DisplayFromDataframe(df, 'close', './tmp/'+value+'_close.png')
    visu.DisplayFromDataframe(df, 'macd', './tmp/'+value+'_macd.png')
    visu.DisplayFromDataframe(df, 'rsi_30', './tmp/'+value+'_rsi_30.png')
    visu.DisplayFromDataframe(df, 'cci_30', './tmp/'+value+'_cci_30.png')
    visu.DisplayFromDataframe(df, 'williams_%r', './tmp/'+value+'_williams%r.png')
    visu.DisplayFromDataframe(df, 'stoch_%k', './tmp/'+value+'_stoch_%k.png')
    visu.DisplayFromDataframe(df, 'stoch_%d', './tmp/'+value+'_stoch_%d.png')
    visu.DisplayFromDataframe(df, 'er', './tmp/'+value+'_er.png')
    visu.DisplayFromDataframe(df, 'stc', './tmp/'+value+'_stc.png')
    visu.DisplayFromDataframe(df, 'sma_5', './tmp/'+value+'_sma5.png')
    visu.DisplayFromDataframe(df, 'sma_10', './tmp/'+value+'_sma10.png')
    visu.DisplayFromDataframe(df, 'sma_15', './tmp/'+value+'_sma15.png')
    visu.DisplayFromDataframe(df, 'sma_20', './tmp/'+value+'_sma20.png')
    visu.DisplayFromDataframe(df, 'ema_10', './tmp/'+value+'_ema10.png')
    visu.DisplayFromDataframe(df, 'ema_20', './tmp/'+value+'_ema20.png')
    visu.DisplayFromDataframe(df, 'ema_50', './tmp/'+value+'_ema50.png')
    visu.DisplayFromDataframe(df, 'atr', './tmp/'+value+'_atr.png')
    visu.DisplayFromDataframe(df, 'adx', './tmp/'+value+'_adx.png')
    visu.DisplayFromDataframe(df, 'roc', './tmp/'+value+'_roc.png')


#
# parse directory cac40
#
def cac40():
    directory = "./lib/data/CAC40/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = fimport.GetDataFrameFromCsv(directory+"/"+filename)
            technical_indicators = ["trend_1d","macd","rsi_30","cci_30","williams_%r","stoch_%k","stoch_%d","er","stc"]
            technical_indicators.extend(["sma_5","sma_10","sma_15","sma_20"])
            technical_indicators.extend(["ema_10","ema_20","ema_50"])
            technical_indicators.extend(["atr","adx","roc"])
            df = findicators.add_technical_indicators(df, technical_indicators)
            trend_ratio, true_positive, true_negative, false_positive, false_negative = findicators.get_trend_info(df)
            value = filename[:len(filename)-4]
            print("{} ({}),{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(value, fimport.cac40[value], trend_ratio, true_positive, true_negative, false_positive, false_negative))




_usage_str = """
Options:
    [--various --stats --cac40]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--various": various()
        elif sys.argv[1] == "--stats" and len(sys.argv) > 2: stats(sys.argv[2])
        elif sys.argv[1] == "--cac40": cac40()
        else: _usage()
    else: _usage()
