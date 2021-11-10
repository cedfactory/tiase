from tiase.ml import classifier,classifier_lstm,classifier_gaussian_process,classifier_mlp,classifier_naive,classifier_naive_bayes,classifier_svc,classifier_xgboost,classifier_decision_tree,hyper_parameters_tuning,meta_classifier
import inspect

import importlib

g_classifiers_library = dict()

class ClassifiersFactory:
    @staticmethod
    def get_classifier(type, params=None):
        if type == "same class":
            return classifier_naive.ClassifierAlwaysSameClass(params)
        elif type == "as previous":
            return classifier_naive.ClassifierAlwaysAsPrevious(params)
        elif type == "decision tree":
            return classifier_decision_tree.ClassifierDecisionTree(params)
        elif type == "gaussian process":
            return classifier_gaussian_process.ClassifierGaussianProcess(params)
        elif type == "mlp":
            return classifier_mlp.ClassifierMLP(params)
        elif type == "gaussian naive bayes":
            return classifier_naive_bayes.ClassifierGaussianNB(params)
        elif type == "svc":
            return classifier_svc.ClassifierSVC(params)
        elif type == "xgboost":
            return classifier_xgboost.ClassifierXGBoost(params)
        elif type == 'lstm1':
            return classifier_lstm.ClassifierLSTM1(params)
        elif type == 'lstm2':
            return classifier_lstm.ClassifierLSTM2(params)
        elif type == 'lstm3':
            return classifier_lstm.ClassifierLSTM3(params)
        elif type == 'lstmhao2020':
            return classifier_lstm.ClassifierLSTMHao2020(params)
        elif type == 'bilstm':
            return classifier_lstm.ClassifierBiLSTM(params)
        elif type == 'cnnbilstm':
            return classifier_lstm.ClassifierCNNBiLSTM(params)
        elif type == 'grid search':
            return hyper_parameters_tuning.HPTGridSearch(params)
        elif type == 'voting':
            return meta_classifier.MetaClassifierVoting(params)

    @staticmethod
    def __get_classifiers_list_from_class(classifier):
        if classifier:
            available_classifiers = []
            if not inspect.isabstract(classifier):
                module = importlib.import_module(classifier.__module__)
                klass = getattr(module, classifier.__qualname__)
                instance = klass()
                print("################")
                print(classifier.__qualname__)
                print(instance.get_name())
                print(instance.get_param_grid())

                available_classifiers = [classifier.__name__]
            subclassifiers = classifier.__subclasses__()
            for subclassifier in subclassifiers:
                available_classifiers.extend(ClassifiersFactory.__get_classifiers_list_from_class(subclassifier))
            return available_classifiers
        else:
            return []

    @staticmethod
    def get_classifiers_list():
        return ClassifiersFactory.__get_classifiers_list_from_class(classifier.Classifier)
