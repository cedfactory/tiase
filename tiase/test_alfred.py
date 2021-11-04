from tiase.fimport import fimport
from tiase import alfred
import numpy as np
import pytest
import os

g_generate_references = False

def compare_dataframes(df1, df2, columns):
    if len(df1.columns) != len(df2.columns):
        print("[compare_dataframes] columns {} vs {}".format(len(df1.columns), len(df2.columns)))
        return False
    for column in columns:
        array1 = df1[column].to_numpy()
        array2 = df2[column].to_numpy()
        if np.allclose(array1, array2) == False:
            print("[compare_dataframes] {}".format(column))
            return False
    return True

class TestAlfred:

    def common_process(self, xml_file, ref_file):
        alfred.execute(xml_file)

        df_generated = fimport.get_dataframe_from_csv("./tmp/out.csv")
        df_generated = df_generated.head(100)

        if g_generate_references:
            df_generated.to_csv(ref_file, float_format='%.8f')
        df_expected = fimport.get_dataframe_from_csv(ref_file)

        assert(compare_dataframes(df_generated, df_expected, df_expected.columns))

    def test_indicators(self):
        self.common_process("./tiase/data/test/alfred_indicators.xml", "./tiase/data/test/alfred_indicators_reference.csv")

    def test_outliers_normalize_stdcutoff(self):
        self.common_process("./tiase/data/test/alfred_outliers.xml", "./tiase/data/test/alfred_outliers_reference.csv")

    def test_outliers_transformation(self):
        self.common_process("./tiase/data/test/alfred_transformation.xml", "./tiase/data/test/alfred_transformation_reference.csv")

    def test_outliers_discretization(self):
        self.common_process("./tiase/data/test/alfred_discretization.xml", "./tiase/data/test/alfred_discretization_reference.csv")

    def test_outliers_classifier(self):
        out_file = "./tmp/lstm1_1.hdf5"
        if os.path.isfile(out_file):
            os.remove(out_file)

        alfred.execute("./tiase/data/test/alfred_classifier.xml")
        
        assert(os.path.isfile(out_file))
        if os.path.isfile(out_file):
            os.remove(out_file) 
