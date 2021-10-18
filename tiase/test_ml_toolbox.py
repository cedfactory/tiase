import pandas as pd
import numpy as np
from tiase.fimport import fimport,synthetic
from tiase.findicators import findicators
from tiase.ml import toolbox
from sklearn import preprocessing
import os

def compare_dataframes(df1, df2, columns):
    if len(df1.columns) != len(df2.columns):
        return False
    for column in columns:
        array1 = df1[column].to_numpy()
        array2 = df2[column].to_numpy()
        if np.allclose(array1, array2) == False:
            return False
    return True


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

    def test_get_train_test_data_list_from_cv_wf_split_dataframe(self):
        y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
        df = synthetic.create_dataframe(y, 0.)
        df = findicators.add_technical_indicators(df, ["target"])
        df = findicators.remove_features(df, ["open","low","high","volume"])
        df.dropna(inplace = True)

        list_df_training, list_df_testing = toolbox.get_train_test_data_list_from_CV_WF_split_dataframe(df, 3)

        assert(len(list_df_training) == 3)
        assert(len(list_df_testing) == 3)
        for idx,df_computed in enumerate(list_df_training + list_df_testing):
            ref_file = "./tiase/data/test/get_train_test_data_list_from_CV_WF_split_dataframe_{}_reference.csv".format(idx)
            #df_computed.to_csv(ref_file)
            df_expected = fimport.get_dataframe_from_csv(ref_file)
            assert(compare_dataframes(df_computed, df_expected, df_expected.columns))
