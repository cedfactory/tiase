from lib.fimport import *
from lib.findicators import *
from lib.featureengineering import fprocessfeature
from lib.fdatapreprocessing import fdataprep
import numpy as np

import pandas as pd
import os

from rich import print,inspect

def check(action):
    print("check \"{}\"".format(action))
    y = synthetic.get_sinusoid(length=5, amplitude=1, frequency=.1, phi=0, height = 0)
    df = synthetic.create_dataframe(y, 0.)
    visu.DisplayFromDataframe(df,"Close", "close.png")

    if action == "missing_values":
        df["Open"][1] = np.nan
        print(df.head())
        df = fdataprep.process_technical_indicators(df, ['missing_values'])
        print(df.head())
        


#
# parse directory cac40
#
def cac40():
    directory = "./lib/data/CAC40/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = fimport.GetDataFrameFromCsv(directory+"/"+filename)
            technical_indicators = ["trend_1d","macd","rsi_30","cci_30","williams_%r","stoch_%k","stoch_%d","er","stc"]
            technical_indicators.extend(["sma_5","sma_10","sma_20"])
            technical_indicators.extend(["ema_5","ema_10","ema_20"])
            technical_indicators.extend(["wma_5","wma_10","wma_20"])
            technical_indicators.extend(["atr","adx","roc"])
            technical_indicators.extend(["simple_rtn", "mom", "target"])

            df = findicators.add_technical_indicators(df, technical_indicators)
            trend_ratio, true_positive, true_negative, false_positive, false_negative = findicators.get_trend_info(df)
            value = filename[:len(filename)-4]
            print("{} ({}),{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(value, fimport.cac40[value], trend_ratio, true_positive, true_negative, false_positive, false_negative))

            df = fdataprep.process_technical_indicators(df, ['missing_values'])
            visu.DisplayHistogramFromDataframe(df, 'simple_rtn', './tmp/' + value + '_close.png')

            df_copy = df.copy()
            df = fdataprep.process_technical_indicators(df, ['outliers_winsorize'])
            visu.DisplayOutliersFromDataframe(df_copy, df, './tmp/' + value + '_outliers.png')

            df = fdataprep.process_technical_indicators(df, ['discretization_unsupervised'])
            df = fprocessfeature.process_features(df, ['correlation_reduction'])
            df = fprocessfeature.process_features(df, ['smote'])
            df = fprocessfeature.process_features(df, ['rfecv_reduction'])
            df = fdataprep.process_technical_indicators(df, ['discretization'])
            df = fprocessfeature.process_features(df, ['correlation_reduction'])
            df = fprocessfeature.process_features(df, ['pca_reduction'])
            df = fdataprep.process_technical_indicators(df, ['discretization'])
            df = fprocessfeature.process_features(df, ['correlation_reduction'])
            df = fdataprep.process_technical_indicators(df, ['discretization_unsupervised'])
            df = fdataprep.process_technical_indicators(df, ['transformation_log'])
            df = fdataprep.process_technical_indicators(df, ['transformation_x2'])
            df = fdataprep.process_technical_indicators(df, ['outliers_stdcutoff'])
            df = fdataprep.process_technical_indicators(df, ['outliers_ema'])
            df = fdataprep.process_technical_indicators(df, ['outliers_mam'])
            df = fdataprep.process_technical_indicators(df, ['outliers_winsorize'])

            break

_usage_str = """
Options:
    [--cac40]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check" and len(sys.argv) > 2: check(sys.argv[2])
        elif sys.argv[1] == "--cac40": cac40()
        else: _usage()
    else: _usage()
