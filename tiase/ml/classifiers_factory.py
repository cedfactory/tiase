from tiase.ml import classifier_lstm,classifier_gaussian_process,classifier_mlp,classifier_naive,classifier_naive_bayes,classifier_svc,classifier_xgboost,classifier_decision_tree

class ClassifiersFactory:
    def get_classifier(name, params, data_splitter):
        if name == "same class":
            return classifier_naive.ClassifierAlwaysSameClass(data_splitter=data_splitter, params=params)
        elif name == "as previous":
            return classifier_naive.ClassifierAlwaysAsPrevious(data_splitter=data_splitter, params=params)
        elif name == "decision tree":
            return classifier_decision_tree.ClassifierDecisionTree(data_splitter=data_splitter, params=params)
        elif name == "gaussian_process":
            return classifier_gaussian_process.ClassifierGaussianProcess(data_splitter=data_splitter, params=params)
        elif name == "mlp":
            return classifier_mlp.ClassifierMLP(data_splitter=data_splitter, params=params)
        elif name == "gaussian_naive_bayes":
            return classifier_naive_bayes.ClassifierGaussianNB(data_splitter=data_splitter, params=params)
        elif name == "svc":
            return classifier_svc.ClassifierSVC(data_splitter=data_splitter, params=params)
        elif name == "xgboost":
            return classifier_xgboost.ClassifierXGBoost(data_splitter=data_splitter, params=params)
        elif name == 'lstm1':
            return classifier_lstm.ClassifierLSTM1(data_splitter=data_splitter, params=params)
        elif name == 'lstm2':
            return classifier_lstm.ClassifierLSTM2(data_splitter=data_splitter, params=params)
        elif name == 'lstm3':
            return classifier_lstm.ClassifierLSTM3(data_splitter=data_splitter, params=params)
        elif name == 'lstmhao2020':
            return classifier_lstm.ClassifierLSTMHao2020(data_splitter=data_splitter, params=params)
        elif name == 'bilstm':
            return classifier_lstm.ClassifierBiLSTM(data_splitter=data_splitter, params=params)
        elif name == 'cnnbilstm':
            return classifier_lstm.ClassifierCNNBiLSTM(data_splitter=data_splitter, params=params)


