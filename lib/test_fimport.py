from lib.fimport import fimport,synthetic
import numpy as np
import pytest

class TestFImport:

    def test_GetDataFrameFromYahoo(self):
        df = fimport.get_dataframe_from_yahoo('AI.PA')
        assert(df.at["2000-01-03",'Close'] == pytest.approx(18.313592, 0.00001))

    def test_synthetic_data_constant(self):
        expected_data = [5.5, 5.5, 5.5, 5.5, 5.5]
        y = synthetic.get_constant(amplitude=5.5, length = 5)

        np.testing.assert_allclose(y, expected_data, 0.00001)

    def test_synthetic_data_linear(self):
        expected_data = [3.,  3.5, 4.,  4.5, 5.,  5.5, 6.,  6.5, 7.,  7.5]
        y = synthetic.get_linear(a=.5, b=3, length=10)

        np.testing.assert_allclose(y, expected_data, 0.00001)

    def test_synthetic_data_sinusoid(self):
        expected_data = [ 0.      ,  0.198669,  0.389418,  0.564642,  0.717356,  0.841471,
               0.932039,  0.98545 ,  0.999574,  0.973848,  0.909297,  0.808496,
               0.675463,  0.515501,  0.334988,  0.14112 , -0.058374, -0.255541,
              -0.44252 , -0.611858]
        y = synthetic.get_sinusoid(length=20, amplitude=1, frequency=.2, phi=0, height = 0)

        np.testing.assert_allclose(y, expected_data, 0.00001)
