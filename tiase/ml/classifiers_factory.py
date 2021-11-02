from tiase.ml import classifier_lstm,classifier_gaussian_process,classifier_mlp,classifier_naive,classifier_naive_bayes,classifier_svc,classifier_xgboost,classifier_decision_tree,hyper_parameters_tuning

class ClassifiersFactory:
    @staticmethod
    def get_classifier(type, params, data_splitter):
        if type == "same class":
            return classifier_naive.ClassifierAlwaysSameClass(data_splitter=data_splitter, params=params)
        elif type == "as previous":
            return classifier_naive.ClassifierAlwaysAsPrevious(data_splitter=data_splitter, params=params)
        elif type == "decision tree":
            return classifier_decision_tree.ClassifierDecisionTree(data_splitter=data_splitter, params=params)
        elif type == "gaussian process":
            return classifier_gaussian_process.ClassifierGaussianProcess(data_splitter=data_splitter, params=params)
        elif type == "mlp":
            return classifier_mlp.ClassifierMLP(data_splitter=data_splitter, params=params)
        elif type == "gaussian naive bayes":
            return classifier_naive_bayes.ClassifierGaussianNB(data_splitter=data_splitter, params=params)
        elif type == "svc":
            return classifier_svc.ClassifierSVC(data_splitter=data_splitter, params=params)
        elif type == "xgboost":
            return classifier_xgboost.ClassifierXGBoost(data_splitter=data_splitter, params=params)
        elif type == 'lstm1':
            return classifier_lstm.ClassifierLSTM1(data_splitter=data_splitter, params=params)
        elif type == 'lstm2':
            return classifier_lstm.ClassifierLSTM2(data_splitter=data_splitter, params=params)
        elif type == 'lstm3':
            return classifier_lstm.ClassifierLSTM3(data_splitter=data_splitter, params=params)
        elif type == 'lstmhao2020':
            return classifier_lstm.ClassifierLSTMHao2020(data_splitter=data_splitter, params=params)
        elif type == 'bilstm':
            return classifier_lstm.ClassifierBiLSTM(data_splitter=data_splitter, params=params)
        elif type == 'cnnbilstm':
            return classifier_lstm.ClassifierCNNBiLSTM(data_splitter=data_splitter, params=params)
        elif type == 'grid search':
            return hyper_parameters_tuning.HPTGridSearch(data_splitter=data_splitter, params=params)


