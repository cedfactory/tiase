#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from lib.fimport import *
from lib.findicators import *


class TestIndicators:
    def test_number_colums(self):
        df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/_AI.PA.csv")
        print(list(df.columns))
        assert(len(list(df.columns)) == 6)
        df = findicators.add_technical_indicators(df)
        assert(len(list(df.columns)) == 10)

