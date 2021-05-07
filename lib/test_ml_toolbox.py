import pandas as pd
import numpy as np
from lib.fimport import *
from lib.ml import toolbox

class TestMlToolbox:

    def test_get_train_test_data(self):
        data = [[10,1000], [2,500], [3,600], [2,800], [2,500], [0,700], [1,800], [3,500], [5,900], [7,1000]]
        df = pd.DataFrame(data, columns = ['indicator', 'Adj Close'])
        df.index.name = 'Date'
        X_train, y_train, X_test, y_test, y_normaliser = toolbox.get_train_test_data(df, 2, 0.6)

        X_train_expected = np.array([[[1., 1. ], [0.2 ,0. ]], [[0.2, 0. ], [0.3, 0.2]], [[0.3, 0.2], [0.2, 0.6]], [[0.2, 0.6], [0.2, 0.]]])
        np.testing.assert_allclose(X_train, X_train_expected, 0.00001)

        y_train_expected = np.array([[0.3], [0.2], [0.2], [0. ]])
        np.testing.assert_allclose(y_train, y_train_expected, 0.00001)

  
         

