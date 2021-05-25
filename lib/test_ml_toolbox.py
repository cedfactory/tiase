import pandas as pd
import numpy as np
from lib.fimport import *
from lib.ml import toolbox
from sklearn import preprocessing
import os

class TestMlToolbox:

    def test_get_train_test_data(self):
        data = [[10,1000], [2,500], [3,600], [2,800], [2,500], [0,700], [1,800], [3,500], [5,900], [7,1000]]
        df = pd.DataFrame(data, columns = ['indicator', 'adj_close'])
        df.index.name = 'Date'
        X_train, y_train, X_test, y_test, y_normaliser = toolbox.get_train_test_data_from_dataframe0(df, 2, 0.6)

        X_train_expected = np.array([[[1., 1. ], [0.2 ,0. ]], [[0.2, 0. ], [0.3, 0.2]], [[0.3, 0.2], [0.2, 0.6]], [[0.2, 0.6], [0.2, 0.]]])
        np.testing.assert_allclose(X_train, X_train_expected, 0.00001)

        y_train_expected = np.array([[0.3], [0.2], [0.2], [0. ]])
        np.testing.assert_allclose(y_train, y_train_expected, 0.00001)


    def test_serialization(self):
        data = [[10,1000], [2,500], [3,600], [2,800], [2,500], [0,700], [1,800], [3,500], [5,900], [7,1000]]
        normalizer = preprocessing.MinMaxScaler()

        normalizer.fit(data)
        print(normalizer)

        toolbox.serialize(normalizer, "./tmp/normalizer.gz")
        normalizer_loaded = toolbox.deserialize("./tmp/normalizer.gz")
        normalized_data = normalizer_loaded.transform(data)

        expected_data = np.array([[1, 1], [0.2, 0], [0.3, 0.2], [0.2, 0.6], [0.2, 0], [0., 0.4], [0.1, 0.6], [0.3, 0], [0.5, 0.8], [0.7, 1]])
        np.testing.assert_allclose(normalized_data, expected_data, 0.00001)

        # cleaning
        os.remove("./tmp/normalizer.gz")


