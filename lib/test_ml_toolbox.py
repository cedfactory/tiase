import pandas as pd
import numpy as np
from lib.fimport import fimport
from lib.ml import toolbox
from sklearn import preprocessing
import os

class TestMlToolbox:

    def test_get_train_test_data_from_dataframe0(self):
        data = [[10,1000], [2,500], [3,600], [2,800], [2,500], [0,700], [1,800], [3,500], [5,900], [7,1000]]
        df = pd.DataFrame(data, columns = ['indicator', 'adj_close'])
        df.index.name = 'Date'
        x_train, y_train, x_test, y_test, x_normaliser, y_normaliser = toolbox.get_train_test_data_from_dataframe0(df, 2, 'adj_close', 0.6)

        x_train_expected = np.array([[[1., 1. ], [0.2 ,0. ]], [[0.2, 0. ], [0.3, 0.2]], [[0.3, 0.2], [0.2, 0.6]], [[0.2, 0.6], [0.2, 0.]]])
        np.testing.assert_allclose(x_train, x_train_expected, 0.00001)

        y_train_expected = np.array([[0.3], [0.2], [0.2], [0. ]])
        np.testing.assert_allclose(y_train, y_train_expected, 0.00001)

    def test_get_train_test_data_from_dataframe1(self):
        data = [[10,1], [2,1], [3,1], [2,1], [2,0], [0,0], [1,1], [3,0], [5,0], [7,1]]
        df = pd.DataFrame(data, columns = ['value', 'target'])
        df.index.name = 'Date'
        x_train, y_train, x_test, y_test, x_normaliser = toolbox.get_train_test_data_from_dataframe1(df, 2, 'target', 0.6)

        x_train_expected = np.array([[1., 0.2], [0.2, 0.3], [0.3, 0.2], [0.2, 0.2]])
        np.testing.assert_allclose(x_train, x_train_expected, 0.00001)

        y_train_expected = np.array([[1], [1], [0], [0]])
        np.testing.assert_allclose(y_train, y_train_expected, 0.00001)


    def test_get_train_test_data_from_dataframe2(self):
        data =[
            [0., 11., 3., 1],
            [0.1, 10., 3.1, 0],
            [0.2, 9., 3.2, 0],
            [0.3, 8., 3.3, 1],
            [0.4, 7., 3.4, 1],
            [0.5, 6., 3.5, 0],
            [0.6, 5., 3.6, 1],
            [0.7, 4., 3.7, 1],
            [0.8, 3., 3.8, 1],
            [0.9, 2., 3.9, 0],
            [1., 1., 4., 0]
        ]
        df = pd.DataFrame(data, columns = ['A', 'B', 'C', 'output'])
        x_train, y_train, x_test, y_test, x_normaliser = toolbox.get_train_test_data_from_dataframe2(df, 3, 'output', 0.7)

        x_train_expected = np.array([[0., 0.16666667, 0.33333333, 1., 0.83333333, 0.66666667, 0., 0.16666667, 0.33333333],
        [0.16666667, 0.33333333, 0.5, 0.83333333, 0.66666667, 0.5, 0.16666667, 0.33333333, 0.5],
        [0.33333333, 0.5, 0.66666667, 0.66666667, 0.5, 0.33333333, 0.33333333, 0.5, 0.66666667],
        [0.5, 0.66666667, 0.83333333, 0.5, 0.33333333, 0.16666667, 0.5, 0.66666667, 0.83333333]])
        np.testing.assert_allclose(x_train, x_train_expected, 0.00001)

        y_train_expected = np.array([[1], [1], [0], [1]])
        np.testing.assert_allclose(y_train, y_train_expected, 0.00001)

        x_test_expected = np.array([[1.16666667, 1.33333333, 1.5, -0.16666667, -0.33333333, -0.5, 1.16666667, 1.33333333, 1.5]])
        np.testing.assert_allclose(x_test, x_test_expected, 0.00001)

        y_test_expected = np.array([[0]])
        np.testing.assert_allclose(y_test, y_test_expected, 0.00001)


    def test_minmaxscaler_serialization(self):
        data = [[10,1000], [2,500], [3,600], [2,800], [2,500], [0,700], [1,800], [3,500], [5,900], [7,1000]]
        normalizer = preprocessing.MinMaxScaler()
        normalizer.fit(data)

        toolbox.serialize(normalizer, "./tmp/normalizer.gz")
        normalizer_loaded = toolbox.deserialize("./tmp/normalizer.gz")
        normalized_data = normalizer_loaded.transform(data)

        expected_data = np.array([[1, 1], [0.2, 0], [0.3, 0.2], [0.2, 0.6], [0.2, 0], [0., 0.4], [0.1, 0.6], [0.3, 0], [0.5, 0.8], [0.7, 1]])
        np.testing.assert_allclose(normalized_data, expected_data, 0.00001)

        # cleaning
        os.remove("./tmp/normalizer.gz")


    def test_get_classification_threshold(self):
        y_test = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_test_prob = np.array([.7, .6, .4, .8, .5, .4, .2, .7, .3, .3])
        threshold = toolbox.get_classification_threshold(y_test, y_test_prob)
        assert(threshold == .4)
