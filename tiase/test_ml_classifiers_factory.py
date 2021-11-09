from tiase.ml import classifiers_factory
import pytest

class TestClassifiersFactory:

    def test_get_classifiers_list(self):
        classifiers_list = classifiers_factory.ClassifiersFactory.get_classifiers_list()
        expected_list = ['ClassifierLSTM1', 'ClassifierLSTM2', 'ClassifierLSTM3', 'ClassifierLSTMHao2020', 'ClassifierBiLSTM', 'ClassifierCNNBiLSTM', 'ClassifierGaussianProcess', 'ClassifierMLP', 'ClassifierAlwaysSameClass', 'ClassifierAlwaysAsPrevious', 'ClassifierGaussianNB', 'ClassifierSVC', 'ClassifierXGBoost', 'ClassifierDecisionTree', 'HPTGridSearch', 'MetaClassifierVoting']
        assert(len(classifiers_list) == len(expected_list))
        assert(not set(classifiers_list) ^ set(expected_list))