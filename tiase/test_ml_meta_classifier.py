import pandas as pd
import numpy as np
from tiase.fimport import fimport
from tiase.findicators import findicators
from tiase.ml import data_splitter,classifiers_factory,meta_classifier
from tiase.ml import toolbox
import pytest

class TestMlClassifier:

    def test_meta_classifier_voting(self):
        filename = "./tiase/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)

        df = findicators.normalize_column_headings(df)
        df = toolbox.make_target(df, "pct_change", 7)
        df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])
        
        target = "target"
        ds = data_splitter.DataSplitterTrainTestSimple(df, target=target, seq_len=21)
        ds.split(0.7)
        
        classifiers = [
            ("DTC3", classifiers_factory.ClassifiersFactory.get_classifier(type="decision tree", params={'max_depth': 3, 'random_state': 1}, data_splitter=ds)),
            ("SVC", classifiers_factory.ClassifiersFactory.get_classifier("svc", None, ds)),
            ("SVC_poly", classifiers_factory.ClassifiersFactory.get_classifier("svc", {'kernel': 'poly'}, ds))
        ]

        for classifier in classifiers:
            model = classifier[1]
            model.fit()
            '''
            model_analysis = model.get_analysis()
            print("------")
            print("Precision : {:.3f}".format(model_analysis["precision"]))
            print("Recall :    {:.3f}".format(model_analysis["recall"]))
            print("f1_score :  {:.3f}".format(model_analysis["f1_score"]))
            '''

        meta_voting = meta_classifier.MetaClassifierVoting(data_splitter=ds, params={'classifiers':classifiers, 'voting': 'soft'})
        meta_voting.fit()
        metamodel_analysis = meta_voting.get_analysis()
        '''
        print("* Precision : {:.3f}".format(metamodel_analysis["precision"]))
        print("* Recall :    {:.3f}".format(metamodel_analysis["recall"]))
        print("* f1_score :  {:.3f}".format(metamodel_analysis["f1_score"]))
        '''
        assert(metamodel_analysis["precision"] == pytest.approx(0.620, 0.1)) # to investigate
        assert(metamodel_analysis["recall"] == pytest.approx(0.969, 0.1)) # to investigate
        assert(metamodel_analysis["f1_score"] == pytest.approx(0.756, 0.1)) # to investigate