import pandas as pd
import numpy as np
from tiase.fimport import fimport
from tiase.findicators import findicators
from tiase.ml import toolbox,data_splitter,classifier_lstm,classifier_svc,classifier_xgboost,classifier_decision_tree,hyper_parameters_tuning
import pytest

class TestMlHyperParametersTuning:

    def test_HPTGridSearch(self):
        # data        
        filename = "./tiase/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)

        # technical indicators & target
        df = findicators.normalize_column_headings(df)
        df = toolbox.make_target(df, "pct_change", 7)
        df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])

        # data splitter
        ds = data_splitter.DataSplitter(df, target="target", seq_len=21)
        ds.split(0.7)

        # classifier
        classifier = classifier_decision_tree.ClassifierDecisionTree(df.copy(), target="target", data_splitter=ds)
        classifier.build()

        # hyper parameters tuning
        hpt_grid_search = hyper_parameters_tuning.HPTGridSearch(classifier, ds)
        best_params = hpt_grid_search.fit()
        print(best_params)

        assert(best_params["criterion"] == 'entropy')
        assert(best_params["max_depth"] == 2)
        assert(best_params["splitter"] == 'best')

        model_analysis = hpt_grid_search.get_analysis()
        print("Analysis")
        print("Accuracy :  {:.4f}".format(model_analysis["accuracy"]))
        print("Precision : {:.4f}".format(model_analysis["precision"]))
        print("Recall :    {:.4f}".format(model_analysis["recall"]))
        print("f1_score :  {:.4f}".format(model_analysis["f1_score"]))

        assert(model_analysis["accuracy"] == pytest.approx(0.389, 0.01))
        assert(model_analysis["precision"] == pytest.approx(1., 0.01))
        assert(model_analysis["recall"] == pytest.approx(.0175, 0.01))
        assert(model_analysis["f1_score"] == pytest.approx(.0345, 0.01))
