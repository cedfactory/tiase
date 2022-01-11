import pandas as pd
import numpy as np
from tiase.fimport import fimport
from tiase.findicators import findicators
from tiase.fdatapreprocessing import fdataprep
from tiase.ml import data_splitter,classifiers_factory
from tiase import alfred
import pytest

class TestMlHyperParametersTuning:

    def _hpt_grid_search(self, classifiername, param_grid=None):
        # data        
        filename = "./tiase/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)

        # technical indicators & target
        df = findicators.normalize_column_headings(df)
        df = findicators.add_technical_indicators(df, ["target"])
        df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        # data splitter
        ds = data_splitter.DataSplitterTrainTestSimple(df, target="target", seq_len=21)
        ds.split(0.7)

        # classifier
        classifier = classifiers_factory.ClassifiersFactory.get_classifier(classifiername)
        classifier.fit(ds)
        if param_grid != None:
            classifier.param_grid = param_grid

        # hyper parameters tuning
        hpt_grid_search = classifiers_factory.ClassifiersFactory.get_classifier("grid search", {"classifier":classifier})
        best_params = hpt_grid_search.fit(ds)
        print(best_params)

        model_analysis = hpt_grid_search.get_analysis()
        print("Analysis")
        print("Accuracy :  {:.4f}".format(model_analysis["accuracy"]))
        print("Precision : {:.4f}".format(model_analysis["precision"]))
        print("Recall :    {:.4f}".format(model_analysis["recall"]))
        print("f1_score :  {:.4f}".format(model_analysis["f1_score"]))

        return best_params, model_analysis


    def test_HPTGridSearch_decision_tree(self):
        best_params, model_analysis = self._hpt_grid_search("decision tree")

        assert(best_params["criterion"] == 'gini')
        assert(best_params["max_depth"] == 4)
        assert(best_params["splitter"] == 'best')

        assert(model_analysis["accuracy"] == pytest.approx(0.5394, 0.01))
        assert(model_analysis["precision"] == pytest.approx(0.5377, 0.01))
        assert(model_analysis["recall"] == pytest.approx(0.9949, 0.01))
        assert(model_analysis["f1_score"] == pytest.approx(0.6981, 0.01))

    def test_HPTGridSearch_decision_tree_with_alfred(self):
        alfred.execute("./tiase/data/test/alfred_classifier_hyper_parameters_tuning.xml")
 
        hpt_grid_search = classifiers_factory.ClassifiersFactory.get_classifier("grid search")
        hpt_grid_search.load("./tmp/hpt_1")
        model_analysis = hpt_grid_search.get_analysis()
        assert(model_analysis["best_param"]["criterion"] == "gini")
        assert(model_analysis["best_param"]["max_depth"] == 4)
        assert(model_analysis["best_param"]["random_state"] == 1)
        assert(model_analysis["best_param"]["splitter"] == "best")
        assert(model_analysis["accuracy"] == pytest.approx(0.54, 0.01))
        assert(model_analysis["precision"] == pytest.approx(0.54, 0.01))
        assert(model_analysis["recall"] == pytest.approx(0.99, 0.01))
        assert(model_analysis["f1_score"] == pytest.approx(0.70, 0.01))

    def test_HPTGridSearch_lstm1(self):
        best_params, model_analysis = self._hpt_grid_search("lstm1", {'epochs': [5, 10], 'batch_size': [10, 15]})

        assert(best_params["epochs"] == 10)
        assert(best_params["batch_size"] == 10)

        assert(model_analysis["accuracy"] == pytest.approx(0.5353, 0.01))
        assert(model_analysis["precision"] == pytest.approx(0.5353, 0.01))
        assert(model_analysis["recall"] == pytest.approx(1.0000, 0.01))
        assert(model_analysis["f1_score"] == pytest.approx(0.6973, 0.01))


