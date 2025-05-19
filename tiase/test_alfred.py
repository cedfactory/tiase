from tiase.fimport import fimport
from tiase import alfred
import numpy as np
import pytest
import os
import filecmp

g_generate_references = False

# todo : to move into classifier.load
import xml.etree.cElementTree as ET
def read_xml(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    accuracy = float(root.find('accuracy').text)
    precision = float(root.find('precision').text)
    recall = float(root.find('recall').text)
    f1_score = float(root.find('f1_score').text)
    return accuracy, precision, recall, f1_score


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

    def test_import_start_end_download(self):
        self.common_process("./tiase/data/test/alfred_import_start_end_download.xml", "./tiase/data/test/alfred_import_start_end_download_reference.csv")

    def test_import_start_end_load(self):
        self.common_process("./tiase/data/test/alfred_import_start_end_load.xml", "./tiase/data/test/alfred_import_start_end_load_reference.csv")

    def test_indicators(self):
        self.common_process("./tiase/data/test/alfred_indicators.xml", "./tiase/data/test/alfred_indicators_reference.csv")

    def test_outliers_normalize_stdcutoff(self):
        self.common_process("./tiase/data/test/alfred_outliers.xml", "./tiase/data/test/alfred_outliers_reference.csv")

    def test_outliers_transformation(self):
        self.common_process("./tiase/data/test/alfred_transformation.xml", "./tiase/data/test/alfred_transformation_reference.csv")

    def test_outliers_discretization(self):
        self.common_process("./tiase/data/test/alfred_discretization.xml", "./tiase/data/test/alfred_discretization_reference.csv")

    def test_classifier(self):
        out_file = "./tmp/lstm1_1.hdf5"
        if os.path.isfile(out_file):
            os.remove(out_file)

        alfred.execute("./tiase/data/test/alfred_classifier.xml")

        accuracy, precision, recall, f1_score = read_xml("./tmp/lstm1_1.xml")
        assert(accuracy == pytest.approx(0.54, 0.1))
        assert(precision == pytest.approx(0.56, 0.1))
        assert(recall == pytest.approx(0.72, 0.1))
        assert(f1_score == pytest.approx(0.60, 0.1))
       
        assert(os.path.isfile(out_file))
        if os.path.isfile(out_file):
            os.remove(out_file) 

    def test_summary(self):
        classifiers_list = alfred.summary()

        expected_classifiers = {'ClassifierLSTM1', 'ClassifierLSTM2', 'ClassifierLSTM3', 'ClassifierLSTMHao2020', 'ClassifierBiLSTM', 'ClassifierCNNBiLSTM',
                                'ClassifierGaussianProcess', 'ClassifierMLP', 'ClassifierGaussianNB', 'ClassifierXGBoost', 'ClassifierSVC', 'ClassifierDecisionTree',
                                'HPTGridSearch', 'MetaClassifierVoting'}

        assert(classifiers_list != None)
        assert(all(x in classifiers_list for x in expected_classifiers))

    def test_details(self):
        alfred.details_for_value("tiase/data/test/google_stocks_data.csv")
        generated_file = "./tmp/index_google_stocks_data.html"
        expected_file = "tiase/data/test/index_google_stocks_data.html"
        assert(filecmp.cmp(generated_file, expected_file))
        