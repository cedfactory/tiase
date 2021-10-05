from tiar.fimport import fimport,visu
from tiar.findicators import findicators
from tiar.fdatapreprocessing import fdataprep
import pandas as pd
import numpy as np
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

    df = fimport.get_dataframe_from_csv("./tiar/data/CAC40/AI.PA.csv")
    visu.display_from_dataframe(df,"Close","close.png")
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
    df = fimport.get_dataframe_from_yahoo(value)
    print("{} ({})".format(value, name))

    technical_indicators = findicators.get_all_default_technical_indicators()
    df = findicators.add_technical_indicators(df, technical_indicators)
    df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

    trend_ratio_1d = findicators.get_stats_for_trend_up(df, 1)
    trend_ratio_7d = findicators.get_stats_for_trend_up(df, 7)
    trend_ratio_21d = findicators.get_stats_for_trend_up(df, 21)
    print("{} ({});{:.2f};{:.2f};{:.2f}".format(value, name, trend_ratio_1d, trend_ratio_7d, trend_ratio_21d))

    true_positive, true_negative, false_positive, false_negative = findicators.get_stats_on_trend_today_equals_trend_tomorrow(df)
    print("{:.2f},{:.2f},{:.2f},{:.2f}".format(true_positive, true_negative, false_positive, false_negative))

    # format for images
    root = './tmp/'
    prefix = value + '_'

    # simple_rtn & histogram
    simple_rtn = df["simple_rtn"].to_numpy()

    visu.display_histogram_fitted_gaussian(simple_rtn, export_name = root + prefix + "simple_rtn_histogram_gaussian.png")
    visu.display_histogram_from_dataframe(df, "simple_rtn", export_name = root + prefix + "simple_rtn_histogram.png")

    # output index.html
    f = open("./tmp/index_"+value+".html", "w")
    f.write("<html><body>")
    f.write("<center><h1>"+value+" ("+name+")</h1></Center>")

    f.write('<h3>trends</h3>')
    f.write("<p>trend ratio d+1 : {:.2f}%</p>".format(trend_ratio_1d))
    f.write("<p>trend ratio d+7 : {:.2f}%</p>".format(trend_ratio_7d))
    f.write("<p>trend ratio d+21 : {:.2f}%</p>".format(trend_ratio_21d))

    f.write('<h3>simple_rtn</h3>')
    print(simple_rtn)
    f.write('<p>mean : '+str(round(simple_rtn.mean(), 6))+'</p>')
    f.write('<p>toto : '+str(len(simple_rtn[simple_rtn > 0])/len(simple_rtn))+'</p>')

    f.write('<p>mean of positive values : '+str(round(simple_rtn[simple_rtn > 0].mean(), 6))+'</p>')
    f.write('<p>mean of negative values : '+str(round(simple_rtn[simple_rtn < 0].mean(), 6))+'</p>')
    f.write('<p>histogram :<br><img width=25% src=' + prefix + "simple_rtn_histogram_gaussian.png" + ' />')
    f.write('<br>')

    '''
    f.write('Indicators : <br>')
    for column in df.columns:
        imgname = column + '.png'
        visu.display_from_dataframe(df, column, root + prefix + imgname)
        f.write('<img width=50% src=' + prefix + imgname + ' />')
    '''
    f.write("</body></html>")
    f.close()

#
# parse directory cac40
#
def cac40():
    directory = "./tiar/data/CAC40/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            value = filename[:len(filename)-4]
            name = fimport.cac40[value]

            df = fimport.get_dataframe_from_csv(directory+"/"+filename)
            technical_indicators = ["trend_1d","macd","rsi_30","cci_30","williams_%r","stoch_%k","stoch_%d","er","stc"]
            technical_indicators.extend(["sma_5","sma_10","sma_15","sma_20"])
            technical_indicators.extend(["ema_10","ema_20","ema_50"])
            technical_indicators.extend(["atr","adx","roc"])
            df = findicators.add_technical_indicators(df, technical_indicators)
            
            trend_ratio, true_positive, true_negative, false_positive, false_negative = findicators.get_trend_info(df)
            print("{} ({});{:.2f};{:.2f};{:.2f};{:.2f};{:.2f}".format(value, name, trend_ratio, true_positive, true_negative, false_positive, false_negative))



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
