from tiar.fimport import fimport
from tiar import alfred
import numpy as np
import pytest

def compare_dataframes(df1, df2, columns):
    for column in columns:
        array1 = df1[column].to_numpy()
        array2 = df2[column].to_numpy()
        if np.allclose(array1, array2) == False:
            return False
    return True

class TestAlfred:

    def test(self):
        alfred.execute("./tiar/data/test/alfred_import_export.xml")

        df_generated = fimport.get_dataframe_from_csv("./tmp/out.csv")
        df_generated = df_generated.head(10)

        ref_file = "./tiar/data/test/alfred_import_export_reference.csv"
        #df_generated.to_csv(ref_file, float_format='%.8f')
        df_expected = fimport.get_dataframe_from_csv(ref_file)

        assert(compare_dataframes(df_generated, df_expected, df_expected.columns))
