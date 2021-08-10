from lib.fimport import fimport,visu
from lib.findicators import findicators
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

    technical_indicators = findicators.get_all_default_technical_indicators()
    df = findicators.add_technical_indicators(df, technical_indicators)

    trend_ratio_1d = findicators.get_stats_for_trend_up(df, 1)
    trend_ratio_7d = findicators.get_stats_for_trend_up(df, 7)
    trend_ratio_21d = findicators.get_stats_for_trend_up(df, 21)
    print("{} ({});{:.2f};{:.2f};{:.2f}".format(value, name, trend_ratio_1d, trend_ratio_7d, trend_ratio_21d))

    true_positive, true_negative, false_positive, false_negative = findicators.get_stats_on_trend_today_equals_trend_tomorrow(df)
    print("{:.2f},{:.2f},{:.2f},{:.2f}".format(true_positive, true_negative, false_positive, false_negative))

    # output index.html
    f = open("./tmp/index_"+value+".html", "w")
    f.write("<html><body>")
    f.write("<center><h1>"+value+" ("+name+")</h1></Center>")

    f.write("<p>trend ratio d+1 : {:.2f}%</p>".format(trend_ratio_1d))
    f.write("<p>trend ratio d+7 : {:.2f}%</p>".format(trend_ratio_7d))
    f.write("<p>trend ratio d+21 : {:.2f}%</p>".format(trend_ratio_21d))

    for column in df.columns:
        imgname = value+'_'+column+'.png'
        visu.DisplayFromDataframe(df, column, './tmp/'+imgname)
        f.write('<img width=50% src='+imgname+' />')

    f.write("</body></html>")
    f.close()

#
# parse directory cac40
#
def cac40():
    directory = "./lib/data/CAC40/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            value = filename[:len(filename)-4]
            name = fimport.cac40[value]

            df = fimport.GetDataFrameFromCsv(directory+"/"+filename)
            technical_indicators = ["trend_1d","macd","rsi_30","cci_30","williams_%r","stoch_%k","stoch_%d","er","stc"]
            technical_indicators.extend(["sma_5","sma_10","sma_15","sma_20"])
            technical_indicators.extend(["ema_10","ema_20","ema_50"])
            technical_indicators.extend(["atr","adx","roc"])
            df = findicators.add_technical_indicators(df, technical_indicators)
            
            trend_ratio_1d = findicators.get_stats_for_trend_up(df, 1)
            trend_ratio_7d = findicators.get_stats_for_trend_up(df, 7)
            trend_ratio_21d = findicators.get_stats_for_trend_up(df, 21)
            print("{} ({});{:.2f};{:.2f};{:.2f}".format(value, name, trend_ratio_1d, trend_ratio_7d, trend_ratio_21d))



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
