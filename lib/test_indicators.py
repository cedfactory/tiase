#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from lib.fimport import *
from lib.findicators import *


class TestIndicators:
    def test_number_colums(self):
        df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/_AI.PA.csv")
        print(list(df.columns))
        assert(len(list(df.columns)) == 6)
        df = findicators.add_technical_indicators(df, ["macd", "rsi_30", "cci_30", "dx_30"])
        assert(len(list(df.columns)) == 10)
        df = findicators.remove_features(df, ["open", "high", "low"])
        assert(len(list(df.columns)) == 7)

