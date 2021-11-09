import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit

from abc import ABCMeta, abstractmethod

class DataSplitter(metaclass = ABCMeta):

    @abstractmethod
    def __init__(self):
        pass


class DataSplitterTrainTest(DataSplitter):

    @abstractmethod
    def split_train_test(self, features, targets):
        ''' will be defined in the inherited classes '''
        pass

    def split(self, train_size):
        if train_size <= 1:
            split_index = int(self.df.shape[0] * train_size)
        else:
            split_index = train_size

        # separate features and targets
        features = self.df.copy().drop(self.target, axis=1)
        target = pd.DataFrame({'target': self.df[self.target]})

        # split training & testing data
        train_features = features[:split_index]
        train_target = target[:split_index]
        train_target = train_target.reset_index(drop=True)

        test_features = features[split_index:]
        test_target = target[split_index:]
        test_target = test_target.reset_index(drop=True)

        self.normalizer = preprocessing.MinMaxScaler()

        train_normalised_features = self.normalizer.fit_transform(train_features)
        test_normalised_features = self.normalizer.transform(test_features)

        self.X_train, self.y_train = self.split_train_test(train_normalised_features, train_target)
        self.X_test, self.y_test = self.split_train_test(test_normalised_features, test_target)

    def export(self, root="."):
        df_x_train = pd.DataFrame(self.X_train)
        df_y_train = pd.DataFrame(self.y_train)
        df_x_test = pd.DataFrame(self.X_test)
        df_y_test = pd.DataFrame(self.y_test)

        df_x_train.to_csv(root+"/x_train.csv")
        df_y_train.to_csv(root+"/y_train.csv")

        df_x_test.to_csv(root+"/x_test.csv")
        df_y_test.to_csv(root+"/y_test.csv")


class DataSplitterTrainTestWithLag(DataSplitterTrainTest):
    """
    build train & test data from a dataframe
    Warnings :
    - when the X data is computed wih rows \in [ i ... i+seq_len [, y is computed with i+seq_len (next row)
    - normalization is performed
    """

    def __init__(self, df, target, seq_len):
        self.df = df
        self.target = target
        self.seq_len = seq_len

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.normalizer = None

    def split_train_test(self, features, targets):
        n_features = features.shape[1]
        x_train = []
        for i in range(len(features) - self.seq_len):
            seq = []
            for j in range(n_features):
                seq.extend(features[:,j][i : i + self.seq_len].flatten())
            x_train.append(seq)
        x_train = np.array(x_train)

        y_train = np.array([targets["target"][i + self.seq_len].copy() for i in range(len(targets) - self.seq_len)])
        y_train = np.expand_dims(y_train, 1)
        
        return x_train, y_train


class DataSplitterTrainTestSimple(DataSplitterTrainTest):
    '''
    build train & test data from a dataframe
    Warnings :
    - when the X data is computed wih rows \in [ i ... i+seq_len-1 ], y is computed with i+seq_len-1 (same row)
    - normalization is performed
    '''

    def __init__(self, df, target, seq_len):
        self.df = df
        self.target = target
        self.seq_len = seq_len

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.normalizer = None

    def split_train_test(self, features, targets):
        n_features = features.shape[1]
        x_train = []
        for i in range(len(features) - self.seq_len + 1):
            seq = []
            for j in range(n_features):
                seq.extend(features[:,j][i : i + self.seq_len].flatten())
            x_train.append(seq)
        x_train = np.array(x_train)

        y_train = np.array([targets["target"][i + self.seq_len - 1].copy() for i in range(len(targets) - self.seq_len + 1)]).astype(int)
        y_train = np.expand_dims(y_train, 1)
        
        return x_train, y_train


class DataSplitterForCrossValidation(DataSplitter):

    def __init__(self, df, nb_splits, max_train_size=500, test_size=100):
        self.df = df
        self.nb_splits = nb_splits
        self.test_size = test_size
        self.max_train_size = max_train_size

    def split(self):
        tscv = TimeSeriesSplit(gap=0, max_train_size=self.max_train_size, n_splits=self.nb_splits, test_size=self.test_size)

        list_df_training = []
        list_df_testing = []
        for split_index in tscv.split(self.df):
            print("df size: ", len(self.df))
            print("TRAIN:", split_index[0][0]," -> ", split_index[0][len(split_index[0])-1], " Size: ", split_index[0][len(split_index[0])-1] - split_index[0][0])
            print("TEST: ", split_index[1][0], " -> ", split_index[1][len(split_index[1]) - 1], " Size: ", split_index[1][len(split_index[1])-1] - split_index[1][0])
            print(" ")

            train = [-1] * split_index[0][0]
            train.extend(split_index[0].tolist().copy())
            train.extend([-1] * (len(self.df) - len(train)))

            test = [-1] * split_index[1][0]
            test.extend(split_index[1].tolist().copy())
            test.extend([-1] * (len(self.df) - len(test)))

            self.df['train'] = train
            self.df['test']  = test
            df_train = self.df[self.df['train'] != -1]
            df_test = self.df[self.df['test'] != -1]
            self.df.drop(columns=['train'], inplace=True)
            self.df.drop(columns=['test'], inplace=True)
            df_train.drop(columns=['train'], inplace=True)
            df_train.drop(columns=['test'], inplace=True)
            df_test.drop(columns=['test'], inplace=True)
            df_test.drop(columns=['train'], inplace=True)
            list_df_training.append(df_train)
            list_df_testing.append(df_test)

        return list_df_training, list_df_testing
