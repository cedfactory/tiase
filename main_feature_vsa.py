from lib.fimport import fimport
from lib.findicators import findicators,vsa
from lib.featureengineering import fprocessfeature
from lib.fdatapreprocessing import fdataprep

import pandas as pd
import os

def cac40():
    directory = "./lib/data/CAC40/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = fimport.get_dataframe_from_csv(directory+"/"+filename)
            technical_indicators = ["trend_1d","macd","rsi_30","cci_30","williams_%r","stoch_%k","stoch_%d","er","stc"]
            technical_indicators.extend(["sma_5","sma_10","sma_20","ema_5","ema_10","ema_20","wma_5","wma_10","wma_20"])
            technical_indicators.extend(["atr","adx","roc","simple_rtn", "mom", "target"])

            technical_indicators.extend(["vsa"])

            df = findicators.add_technical_indicators(df, technical_indicators)

            df = fdataprep.process_technical_indicators(df, ['missing_values'])
            #df = fdataprep.process_technical_indicators(df, ['outliers_cut_stdcutoff'])
            #df = fdataprep.process_technical_indicators(df, ['drop_ohlcv'])
            #df = fprocessfeature.process_features(df, ['kbest_reduction'])
            print(df.head())
            df = fprocessfeature.process_features(df, ['vsa_reduction'])


_usage_str = """
Options:
    [--cac40]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--cac40": cac40()
        else: _usage()
    else: _usage()