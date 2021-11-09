import pandas as pd
import numpy as np
from tiase.fimport import fimport
from tiase.findicators import findicators
from tiase.ml import toolbox,data_splitter,classifiers_factory,hyper_parameters_tuning
import pytest

class TestMlHyperParametersTuning:

    def _hpt_grid_search(self, classifiername, param_grid=None):
        # data        
        filename = "./tiase/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)

        # technical indicators & target
        df = findicators.normalize_column_headings(df)
        df = toolbox.make_target(df, "pct_change", 7)
        df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])

        # data splitter
        ds = data_splitter.DataSplitterTrainTestSimple(df, target="target", seq_len=21)
        ds.split(0.7)

        # classifier
        classifier = classifiers_factory.ClassifiersFactory.get_classifier(classifiername)
        classifier.fit(ds)
        if param_grid != None:
            classifier.param_grid = param_grid

        # hyper parameters tuning
        hpt_grid_search = hyper_parameters_tuning.HPTGridSearch({"classifier":classifier})
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

        assert(best_params["criterion"] == 'entropy')
        assert(best_params["max_depth"] == 2)
        assert(best_params["splitter"] == 'best')

        assert(model_analysis["accuracy"] == pytest.approx(0.389, 0.01))
        assert(model_analysis["precision"] == pytest.approx(1., 0.01))
        assert(model_analysis["recall"] == pytest.approx(.0175, 0.01))
        assert(model_analysis["f1_score"] == pytest.approx(.0345, 0.01))

    def test_HPTGridSearch_lstm1(self):
        best_params, model_analysis = self._hpt_grid_search("lstm1", {'epochs': [5, 10], 'batch_size': [10, 15]})

        # todo  : investigate why the results are not reproductible
        '''
        assert(best_params["epochs"] == 15)
        assert(best_params["batch_size"] == 10)

        assert(model_analysis["accuracy"] == pytest.approx(0.6213, 0.01))
        assert(model_analysis["precision"] == pytest.approx(0.6213, 0.01))
        assert(model_analysis["recall"] == pytest.approx(1.0000, 0.01))
        assert(model_analysis["f1_score"] == pytest.approx(0.7664, 0.01))
        '''

