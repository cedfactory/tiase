import pandas as pd
import numpy as np
from tiase.ml import data_splitter
import pandas as pd
import os

g_generate_references = False

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

    def get_dataframe(self):
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
        return pd.DataFrame(data, columns = ['A', 'B', 'C', 'target'])

    def test_data_splitter_with_lag(self):
        df = self.get_dataframe()
        ds = data_splitter.DataSplitterTrainTestWithLag(df, target='target', seq_len=3)
        ds.split(0.7)

        x_train_expected = np.array([
            [0., 0.16666667, 0.33333333, 1., 0.83333333, 0.66666667, 0., 0.16666667, 0.33333333],
            [0.16666667, 0.33333333, 0.5, 0.83333333, 0.66666667, 0.5, 0.16666667, 0.33333333, 0.5],
            [0.33333333, 0.5, 0.66666667, 0.66666667, 0.5, 0.33333333, 0.33333333, 0.5, 0.66666667],
            [0.5, 0.66666667, 0.83333333, 0.5, 0.33333333, 0.16666667, 0.5, 0.66666667, 0.83333333]])
        np.testing.assert_allclose(ds.X_train, x_train_expected, 0.00001)

        y_train_expected = np.array([[1], [1], [0], [1]])
        np.testing.assert_allclose(ds.y_train, y_train_expected, 0.00001)

        x_test_expected = np.array([[1.16666667, 1.33333333, 1.5, -0.16666667, -0.33333333, -0.5, 1.16666667, 1.33333333, 1.5]])
        np.testing.assert_allclose(ds.X_test, x_test_expected, 0.00001)

        y_test_expected = np.array([[1]])
        np.testing.assert_allclose(ds.y_test, y_test_expected, 0.00001)

    def test_data_splitter(self):
        df = self.get_dataframe()
        ds = data_splitter.DataSplitterTrainTestSimple(df, target='target', seq_len=3)
        ds.split(0.7)

        x_train_expected = np.array([
            [0., 0.16666667, 0.33333333, 1., 0.83333333, 0.66666667, 0., 0.16666667, 0.33333333],
            [0.16666667, 0.33333333, 0.5, 0.83333333, 0.66666667, 0.5, 0.16666667, 0.33333333, 0.5],
            [0.33333333, 0.5, 0.66666667, 0.66666667, 0.5, 0.33333333, 0.33333333, 0.5, 0.66666667],
            [0.5, 0.66666667, 0.83333333, 0.5, 0.33333333, 0.16666667, 0.5, 0.66666667, 0.83333333],
            [0.66666667, 0.83333333, 1., 0.33333333, 0.16666667, 0., 0.66666667, 0.83333333, 1.]])
        np.testing.assert_allclose(ds.X_train, x_train_expected, 0.00001)

        y_train_expected = np.array([[0], [1], [1], [0], [1]])
        np.testing.assert_allclose(ds.y_train, y_train_expected, 0.00001)

        x_test_expected = np.array([
            [1.16666667, 1.33333333, 1.5, -0.16666667, -0.33333333, -0.5, 1.16666667, 1.33333333, 1.5],
            [1.33333333, 1.5, 1.66666667, -0.33333333, -0.5, -0.66666667, 1.33333333, 1.5, 1.66666667]])
        np.testing.assert_allclose(ds.X_test, x_test_expected, 0.00001)

        y_test_expected = np.array([[0], [1]])
        np.testing.assert_allclose(ds.y_test, y_test_expected, 0.00001)     

    def test_data_splitter_export(self):
        df = self.get_dataframe()
        ds = data_splitter.DataSplitterTrainTestSimple(df, target='target', seq_len=3)
        ds.split(0.7)

        root_tmp = "./tmp/"
        x_train = root_tmp+"x_train.csv"
        x_test = root_tmp+"x_test.csv"
        y_train = root_tmp+"y_train.csv"
        y_test = root_tmp+"y_test.csv"

        root_ref = "./tiase/data/test/"
        ref_x_train = root_ref+"ref_x_train.csv"
        ref_x_test = root_ref+"ref_x_test.csv"
        ref_y_train = root_ref+"ref_y_train.csv"
        ref_y_test = root_ref+"ref_y_test.csv"
        if g_generate_references:
            ds.export("./tmp/")
            os.rename(x_train, ref_x_train)
            os.rename(x_test, ref_x_test)
            os.rename(y_train, ref_y_train)
            os.rename(y_test, ref_y_test)

        ds.export(root_tmp)
        for (file_generated, file_expected) in [(x_train,ref_x_train), (x_test,ref_x_test), (y_train,ref_y_train), (y_test,ref_y_test)]:
            df_generated = pd.read_csv(file_generated)
            df_expected = pd.read_csv(file_expected)
            assert(compare_dataframes(df_generated, df_expected))

    def test_data_splitter_for_cross_validation(self):
        # Prepare the data
        df = self.get_dataframe()

        # instantiate the data splitter
        ds = data_splitter.DataSplitterForCrossValidation(df, nb_splits=3, test_size=2)
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
