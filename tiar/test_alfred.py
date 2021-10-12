from tiar.fimport import fimport
from tiar import alfred
import numpy as np
import pytest

g_generate_references = False

def compare_dataframes(df1, df2, columns):
    if len(df1.columns) != len(df2.columns):
        return False
    for column in columns:
        array1 = df1[column].to_numpy()
        array2 = df2[column].to_numpy()
        if np.allclose(array1, array2) == False:
            return False
    return True

class TestAlfred:

    def test_indicators(self):
        alfred.execute("./tiar/data/test/alfred_indicators.xml")

        df_generated = fimport.get_dataframe_from_csv("./tmp/out.csv")
        df_generated = df_generated.head(100)

        ref_file = "./tiar/data/test/alfred_indicators_reference.csv"
        if g_generate_references:
            df_generated.to_csv(ref_file, float_format='%.8f')
        df_expected = fimport.get_dataframe_from_csv(ref_file)

        assert(compare_dataframes(df_generated, df_expected, df_expected.columns))


    def test_outliers_normalize_stdcutoff(self):
        alfred.execute("./tiar/data/test/alfred_outliers.xml")

        df_generated = fimport.get_dataframe_from_csv("./tmp/out.csv")
        df_generated = df_generated.head(100)

        ref_file = "./tiar/data/test/alfred_outliers_reference.csv"
        if g_generate_references:
            df_generated.to_csv(ref_file, float_format='%.8f')
        df_expected = fimport.get_dataframe_from_csv(ref_file)

        assert(compare_dataframes(df_generated, df_expected, df_expected.columns))


    def test_outliers_transformation(self):
        alfred.execute("./tiar/data/test/alfred_transformation.xml")

        df_generated = fimport.get_dataframe_from_csv("./tmp/out.csv")
        df_generated = df_generated.head(100)

        ref_file = "./tiar/data/test/alfred_transformation_reference.csv"
        if g_generate_references:
            df_generated.to_csv(ref_file, float_format='%.8f')
        df_expected = fimport.get_dataframe_from_csv(ref_file)

        assert(compare_dataframes(df_generated, df_expected, df_expected.columns))

    def test_outliers_discretization(self):
        alfred.execute("./tiar/data/test/alfred_discretization.xml")

        df_generated = fimport.get_dataframe_from_csv("./tmp/out.csv")
        df_generated = df_generated.head(100)

        ref_file = "./tiar/data/test/alfred_discretization_reference.csv"
        if g_generate_references:
            df_generated.to_csv(ref_file, float_format='%.8f')
        df_expected = fimport.get_dataframe_from_csv(ref_file)

        assert(compare_dataframes(df_generated, df_expected, df_expected.columns))
