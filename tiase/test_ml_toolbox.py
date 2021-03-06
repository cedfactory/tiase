import pandas as pd
import numpy as np
from tiase.findicators import findicators
from tiase.ml import toolbox
from sklearn import preprocessing
import os

class TestMlToolbox:

    def test_minmaxscaler_serialization(self):
        data = [[10,1000], [2,500], [3,600], [2,800], [2,500], [0,700], [1,800], [3,500], [5,900], [7,1000]]
        normalizer = preprocessing.MinMaxScaler()
        normalizer.fit(data)

        filename = "./tmp/normalizer.gz"
        toolbox.serialize(normalizer, filename)
        normalizer_loaded = toolbox.deserialize(filename)
        normalized_data = normalizer_loaded.transform(data)

        expected_data = np.array([[1, 1], [0.2, 0], [0.3, 0.2], [0.2, 0.6], [0.2, 0], [0., 0.4], [0.1, 0.6], [0.3, 0], [0.5, 0.8], [0.7, 1]])
        np.testing.assert_allclose(normalized_data, expected_data, 0.00001)

        # cleaning
        os.remove(filename)


    def test_get_classification_threshold_unknown(self):
        y_test = np.array(([1], [1], [1], [1], [1], [0], [0], [0], [0], [0]))
        y_test_prob = np.array(([.7], [.6], [.4], [.8], [.5], [.4], [.2], [.7], [.3], [.3]))
        threshold, y_test_pred = toolbox.get_classification_threshold("foobar", y_test, y_test_prob)
        assert(threshold == -1.)
        assert(len(y_test_pred) == 0)

    def test_get_classification_threshold_naive(self):
        y_test = np.array(([1], [1], [1], [1], [1], [0], [0], [0], [0], [0]))
        y_test_prob = np.array(([.7], [.6], [.4], [.8], [.5], [.4], [.2], [.7], [.3], [.3]))
        threshold, y_test_pred = toolbox.get_classification_threshold("naive", y_test, y_test_prob)
        assert(threshold == .5)
        y_test_pred_expected = np.array(([1], [1], [0], [1], [0], [0], [0], [1], [0], [0]))
        np.testing.assert_allclose(y_test_pred, np.array(y_test_pred_expected), 0.00001)

    def test_get_classification_threshold_best_accuracy_score(self):
        y_test = np.array(([1], [1], [1], [1], [1], [0], [0], [0], [0], [0]))
        y_test_prob = np.array(([.7], [.6], [.4], [.8], [.5], [.4], [.2], [.7], [.3], [.3]))
        threshold, y_test_pred = toolbox.get_classification_threshold("best_accuracy_score", y_test, y_test_prob)
        assert(threshold == .4)
        y_test_pred_expected = np.array(([1], [1], [0], [1], [1], [0], [0], [1], [0], [0]))
        np.testing.assert_allclose(y_test_pred, np.array(y_test_pred_expected), 0.00001)


    def test_is_multiclass(self):
        multiclass = toolbox.is_multiclass(np.array([1, 1, 2, 2, 1]))
        assert(multiclass == False)

        multiclass = toolbox.is_multiclass(np.array([1, 1, 2, 2, 3]))
        assert(multiclass == True)

        df = pd.DataFrame([[1], [0], [1], [1], [0], [1], [0], [1]], columns = ['target'])
        multiclass = toolbox.is_multiclass(df, target="target")
        assert(multiclass == False)

        df = pd.DataFrame([[1], [0], [1], [2], [0], [1], [0], [1]], columns = ['target'])
        multiclass = toolbox.is_multiclass(df, target="target")
        assert(multiclass == True)

    def test_get_n_classes(self):
        n_classes = toolbox.get_n_classes(np.array([1, 1, 2, 2, 1]))
        assert(n_classes == 2)

        n_classes = toolbox.get_n_classes(np.array([1, 1, 2, 2, 3]))
        assert(n_classes == 3)

        df = pd.DataFrame([[1], [0], [1], [1], [0], [1], [0], [1]], columns = ['target'])
        n_classes = toolbox.get_n_classes(df, target="target")
        assert(n_classes == 2)

        df = pd.DataFrame([[1], [0], [1], [2], [0], [1], [0], [1]], columns = ['target'])
        n_classes = toolbox.get_n_classes(df, target="target")
        assert(n_classes == 3)
