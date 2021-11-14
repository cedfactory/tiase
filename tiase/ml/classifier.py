from abc import ABCMeta, abstractmethod

import pandas as pd
from . import data_splitter
from . import toolbox

def get_pred_and_prob_with_predict_pred_and_predict_proba(model, data_splitter):
    y_test_pred = model.predict(data_splitter.X_test)
    y_test_prob = model.predict_proba(data_splitter.X_test)
    y_test_prob = y_test_prob[:, 1]
    return y_test_pred, y_test_prob

def set_train_test_data(dataframe, seq_len, split_index, target):
    # split the data
    ds = data_splitter.DataSplitterTrainTestSimple(dataframe, target, seq_len)
    ds.split(split_index)
    y_train = ds.y_train.astype(int)
    y_test = ds.y_test.astype(int)
    return ds.X_train, y_train, ds.X_test, y_test, ds.normalizer

class Classifier(metaclass = ABCMeta):
    
    def __init__(self, params = None):
        self.seq_len = 21
        if params:
            self.seq_len = params.get("seq_len", self.seq_len)

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_param_grid(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def fit(self, data_splitter):
        pass

    @abstractmethod
    def get_analysis(self):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass

    def evaluate_cross_validation(self, ds, target, debug=False):
        dump_data_to_df = ["tic", "train_size", "test_size", "sum_pred", "threshold",
            "pred_pos_rate","accuracy", "precision", "recall", "f1_score",
            "pred_pos_rate","accuracy", "precision", "recall", "f1_score",
            "pred_pos_rate_0","accuracy_0", "precision_0", "recall_0", "f1_score_0",
            "pred_pos_rate_1","accuracy_1", "precision_1", "recall_1", "f1_score_1",
            "pred_pos_rate_2","accuracy_2", "precision_2", "recall_2", "f1_score_2"]

        results = {}
        results["accuracies"] = []
        results["average_accuracy"] = 0

        lst_cv_splits = ds.split()

        # data for debug
        dump_analysis = pd.DataFrame(columns=dump_data_to_df)
        dump_predictions = pd.DataFrame(columns=['iteration','y_test','y_test_prob','y_test_pred'])
       
        for index_lst_split in range(len(lst_cv_splits[0])):
            split_index = len(lst_cv_splits[0][index_lst_split])
            frames = [lst_cv_splits[0][index_lst_split], lst_cv_splits[1][index_lst_split]]
            df_cv = pd.concat(frames)

            self.X_train, self.y_train, self.X_test, self.y_test, self.x_normaliser = set_train_test_data(df_cv, self.seq_len, split_index, target)

            ds_tmp=data_splitter.DataSplitterCustom()
            ds_tmp.X_train = self.X_train
            ds_tmp.y_train = self.y_train
            ds_tmp.X_test = self.X_test
            ds_tmp.y_test = self.y_test
            ds_tmp.normalizer = self.x_normaliser
            self.fit(ds_tmp)

            current_analysis = self.get_analysis()
            results["accuracies"].append(current_analysis["accuracy"])

            if debug:
                iteration = index_lst_split

                # for dump predictions
                df_tmp = pd.DataFrame(columns=['iteration','y_test','y_test_prob','y_test_pred'])
                df_tmp['iteration'] = [iteration] * len(self.y_test_prob)
                df_tmp['y_test_prob'] = [x[0] for x in self.y_test_prob]
                df_tmp['y_test'] = [x[0] for x in self.y_test]
                df_tmp['y_test_pred'] = [x[0] for x in self.y_test_pred]

                dump_predictions = pd.concat([dump_predictions, df_tmp], axis=0)
                dump_predictions.reset_index(drop=True, inplace=True)

                # for dump analysis
                current_analysis = [iteration,
                                len(self.y_train),
                                round(current_analysis['test_size'],2),
                                self.y_test_pred.sum(),
                                self.threshold,

                                round(current_analysis['pred_pos_rate'],2),
                                round(current_analysis['accuracy'],2),
                                round(current_analysis['precision'],2),
                                round(current_analysis['recall'],2),
                                round(current_analysis['f1_score'],2),

                                round(current_analysis['pred_pos_rate_0'], 2),
                                round(current_analysis['accuracy_0'], 2),
                                round(current_analysis['precision_0'], 2),
                                round(current_analysis['recall_0'], 2),
                                round(current_analysis['f1_score_0'], 2),

                                round(current_analysis['pred_pos_rate_1'], 2),
                                round(current_analysis['accuracy_1'], 2),
                                round(current_analysis['precision_1'], 2),
                                round(current_analysis['recall_1'], 2),
                                round(current_analysis['f1_score_1'], 2),

                                round(current_analysis['pred_pos_rate_2'], 2),
                                round(current_analysis['accuracy_2'], 2),
                                round(current_analysis['precision_2'], 2),
                                round(current_analysis['recall_2'], 2),
                                round(current_analysis['f1_score_2'], 2),

                                round(current_analysis['pred_pos_rate_3'], 2),
                                round(current_analysis['accuracy_3'], 2),
                                round(current_analysis['precision_3'], 2),
                                round(current_analysis['recall_3'], 2),
                                round(current_analysis['f1_score_3'], 2)
                                ]
                dump_analysis = toolbox.add_row_to_df(dump_analysis, current_analysis)
  
        if debug:
            dump_predictions.to_csv('./tmp/cross_validation_results.csv')
            dump_analysis.to_csv('./tmp/cross_validation_analysis.csv')

        results["average_accuracy"] = sum(results["accuracies"]) / len(results["accuracies"])
        return results
