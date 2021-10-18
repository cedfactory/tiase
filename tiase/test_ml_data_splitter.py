import pandas as pd
import numpy as np
from tiase.fimport import fimport,synthetic
from tiase.findicators import findicators
from tiase.ml import data_splitter

def compare_dataframes(df1, df2):
    if len(df1.columns) != len(df2.columns):
        print("{} vs {}".format(len(df1.columns), len(df2.columns)))
        return False
    for column in df1.columns:
        array1 = df1[column].to_numpy()
        array2 = df2[column].to_numpy()
        if np.allclose(array1, array2) == False:
            return False
    return True

class TestMlDataSplitter:

    def test_data_splitter_with_lag(self):
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
        df = pd.DataFrame(data, columns = ['A', 'B', 'C', 'target'])

        ds = data_splitter.DataSplitterWithLag(df, target='target', seq_len=3)
        ds.split(0.7)

        x_train_expected = np.array([[0., 0.16666667, 0.33333333, 1., 0.83333333, 0.66666667, 0., 0.16666667, 0.33333333],
        [0.16666667, 0.33333333, 0.5, 0.83333333, 0.66666667, 0.5, 0.16666667, 0.33333333, 0.5],
        [0.33333333, 0.5, 0.66666667, 0.66666667, 0.5, 0.33333333, 0.33333333, 0.5, 0.66666667],
        [0.5, 0.66666667, 0.83333333, 0.5, 0.33333333, 0.16666667, 0.5, 0.66666667, 0.83333333]])
        np.testing.assert_allclose(ds.X_train, x_train_expected, 0.00001)

        y_train_expected = np.array([[1], [1], [0], [1]])
        np.testing.assert_allclose(ds.y_train, y_train_expected, 0.00001)

        x_test_expected = np.array([[1.16666667, 1.33333333, 1.5, -0.16666667, -0.33333333, -0.5, 1.16666667, 1.33333333, 1.5]])
        np.testing.assert_allclose(ds.X_test, x_test_expected, 0.00001)

        y_test_expected = np.array([[0]])
        np.testing.assert_allclose(ds.y_test, y_test_expected, 0.00001)

    def test_data_splitter(self):
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
            [1., 1., 4., 1]
        ]
        df = pd.DataFrame(data, columns = ['A', 'B', 'C', 'target'])

        ds = data_splitter.DataSplitter(df, target='target', seq_len=3)
        ds.split(0.7)

        X_train_expected = np.array([[0., 0.16666667, 0.33333333, 1., 0.83333333, 0.66666667, 0., 0.16666667, 0.33333333],
        [0.16666667, 0.33333333, 0.5, 0.83333333, 0.66666667, 0.5, 0.16666667, 0.33333333, 0.5],
        [0.33333333, 0.5, 0.66666667, 0.66666667, 0.5, 0.33333333, 0.33333333, 0.5, 0.66666667],
        [0.5, 0.66666667, 0.83333333, 0.5, 0.33333333, 0.16666667, 0.5, 0.66666667, 0.83333333],
        [0.66666667, 0.83333333, 1., 0.33333333, 0.16666667, 0., 0.66666667, 0.83333333, 1.]])
        np.testing.assert_allclose(ds.X_train, X_train_expected, 0.00001)

        y_train_expected = np.array([[0], [1], [1], [0], [1]])
        np.testing.assert_allclose(ds.y_train, y_train_expected, 0.00001)

        X_test_expected = np.array([[1.16666667, 1.33333333, 1.5, -0.16666667, -0.33333333, -0.5, 1.16666667, 1.33333333, 1.5],
        [1.33333333, 1.5, 1.66666667, -0.33333333, -0.5, -0.66666667, 1.33333333, 1.5, 1.66666667]])
        np.testing.assert_allclose(ds.X_test, X_test_expected, 0.00001)

        y_test_expected = np.array([[0], [1]])
        np.testing.assert_allclose(ds.y_test, y_test_expected, 0.00001)

    def test_data_splitter_for_cross_validation(self):
        # Prepare the data
        data = [
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
            [1., 1., 4., 1]
        ]
        df = pd.DataFrame(data, columns = ['A', 'B', 'C', 'target'])

        # instantiate the data splitter
        ds = data_splitter.DataSplitterForCrossValidation(df, target='target', nb_splits=3, test_size=2)
        list_df_training, list_df_testing = ds.split()

        #expectations
        data_training1 = [
            [0., 11., 3., 1],
            [0.1, 10., 3.1, 0],
            [0.2, 9., 3.2, 0],
            [0.3, 8., 3.3, 1],
            [0.4, 7., 3.4, 1]]
        df_training1 = pd.DataFrame(data_training1, columns = ['A', 'B', 'C', 'target'])
        data_training2 = [
            [0., 11., 3., 1],
            [0.1, 10., 3.1, 0],
            [0.2, 9., 3.2, 0],
            [0.3, 8., 3.3, 1],
            [0.4, 7., 3.4, 1],
            [0.5, 6., 3.5, 0],
            [0.6, 5., 3.6, 1]]
        df_training2 = pd.DataFrame(data_training2, columns = ['A', 'B', 'C', 'target'])
        data_training3 = [
            [0., 11., 3., 1],
            [0.1, 10., 3.1, 0],
            [0.2, 9., 3.2, 0],
            [0.3, 8., 3.3, 1],
            [0.4, 7., 3.4, 1],
            [0.5, 6., 3.5, 0],
            [0.6, 5., 3.6, 1],
            [0.7, 4., 3.7, 1],
            [0.8, 3., 3.8, 1]]
        df_training3 = pd.DataFrame(data_training3, columns = ['A', 'B', 'C', 'target'])
        expected_list_df_training = [df_training1, df_training2, df_training3]
        assert(len(expected_list_df_training) == len(list_df_training))
        for i in range(len(expected_list_df_training)):
            assert(compare_dataframes(expected_list_df_training[i], list_df_training[i]))

        data_testing1 = [
            [0.5, 6., 3.5, 0],
            [0.6, 5., 3.6, 1]]
        df_testing1 = pd.DataFrame(data_testing1, columns = ['A', 'B', 'C', 'target'])
        data_testing2 = [
            [0.7, 4., 3.7, 1],
            [0.8, 3., 3.8, 1]]
        df_testing2 = pd.DataFrame(data_testing2, columns = ['A', 'B', 'C', 'target'])
        data_testing3 = [
            [0.9, 2., 3.9, 0],
            [1., 1., 4., 1]]
        df_testing3 = pd.DataFrame(data_testing3, columns = ['A', 'B', 'C', 'target'])
        expected_list_df_testing = [df_testing1, df_testing2, df_testing3]
        assert(len(expected_list_df_testing) == len(list_df_testing))
        for i in range(len(expected_list_df_testing)):
            assert(compare_dataframes(expected_list_df_testing[i], list_df_testing[i]))

        print("list_df_training")
        print(list_df_training)
        print("list_df_testing")
        print(list_df_testing)