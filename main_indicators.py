from tiase.fimport import fimport,visu
from tiase.findicators import findicators
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

    df = fimport.get_dataframe_from_csv("./tiase/data/CAC40/AI.PA.csv")
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

#
# parse directory cac40
#
def cac40():
    directory = "./tiase/data/CAC40/"
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
    [--various --cac40]
"""


def _usage():
    print(_usage_str)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--various":
            various()
        elif sys.argv[1] == "--cac40":
            cac40()
        else:
            _usage()
    else:
        _usage()
