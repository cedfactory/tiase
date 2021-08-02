import pandas as pd
import numpy as np
from lib.fimport import *
from lib.findicators import *
from lib.fdatapreprocessing import *
import pytest

class TestDataProcess:

    def get_dataframe(self):
        y = synthetic.get_sinusoid(length=5, amplitude=1, frequency=.1, phi=0, height = 0)
        df = synthetic.create_dataframe(y, 0.)
        return df

    def test_missing_values(self):
        df = self.get_dataframe()
        df["Open"][1] = np.nan

        assert(df.shape[0] == 5)
        df = fdataprep.process_technical_indicators(df, ['missing_values'])
        assert(df.shape[0] == 4)
