from ..findicators import findicators
from . import toolbox,analysis,classifier

from rich import print,inspect

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, ReLU, BatchNormalization, AveragePooling1D, Conv1D, concatenate, Concatenate, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import backend
from keras import Model
# Functional API : https://keras.io/guides/functional_api/
from tensorflow import keras
from tensorflow.keras import layers

dump_data_to_df = ["tic", "train_size", "test_size", "sum_pred", "threshold",
                   "pred_pos_rate","accuracy", "precision", "recall", "f1_score",
                   "pred_pos_rate_0","accuracy_0", "precision_0", "recall_0", "f1_score_0",
                   "pred_pos_rate_1","accuracy_1", "precision_1", "recall_1", "f1_score_1",
                   "pred_pos_rate_2","accuracy_2", "precision_2", "recall_2", "f1_score_2",
                   "pred_pos_rate_3","accuracy_3", "precision_3", "recall_3", "f1_score_3"]
#
# Simple LSTM for Sequence Classification
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
#
class ClassifierLSTM(classifier.Classifier):
    def __init__(self, dataframe, data_splitter, params = None):
        super().__init__(dataframe, data_splitter, params)

        self.epochs = 170
        self.batch_size = 10
        if params:
            self.epochs = params.get("epochs", self.epochs)
            self.batch_size = params.get("batch_size", self.batch_size)

        if isinstance(self.epochs, str):
            self.epochs = int(self.epochs)

        self.X_train = self.data_splitter.X_train
        self.y_train = self.data_splitter.y_train
        self.X_test = self.data_splitter.X_test
        self.y_test = self.data_splitter.y_test
        self.x_normaliser = self.data_splitter.normalizer
        self.input_size = self.X_train.shape[1]

        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))


    def fit(self):
        # create the model
        tf.random.set_seed(20)
        np.random.seed(10)

        self.build()

        #print(self.model.summary())
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        # Final evaluation of the model
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    def get_analysis(self):
        self.y_test_prob = self.model.predict(self.X_test)

        self.threshold, self.y_test_pred = toolbox.get_classification_threshold("naive", self.y_test, self.y_test_prob)

        self.analysis = analysis.classification_analysis(self.X_test, self.y_test, self.y_test_pred, self.y_test_prob)
        self.analysis["history"] = getattr(self.model, "history", None)
        return self.analysis

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename+".hdf5")

    def save(self, filename):
        self.model.save(filename+".hdf5")

    def evaluate_cross_validation(self, ds, target, debug=False):
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

            self.X_train, self.y_train, self.X_test, self.y_test, self.x_normaliser = classifier.set_train_test_data(df_cv, self.seq_len, split_index, target)
            self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
            self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))

            # create the model
            tf.random.set_seed(20)
            np.random.seed(10)

            self.build()
            self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=self.epochs, batch_size=10, verbose=0)

            self.get_analysis()
            results["accuracies"].append(self.analysis["accuracy"])

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
                                round(self.analysis['test_size'],2),
                                self.y_test_pred.sum(),
                                self.threshold,

                                round(self.analysis['pred_pos_rate'],2),
                                round(self.analysis['accuracy'],2),
                                round(self.analysis['precision'],2),
                                round(self.analysis['recall'],2),
                                round(self.analysis['f1_score'],2),

                                round(self.analysis['pred_pos_rate_0'], 2),
                                round(self.analysis['accuracy_0'], 2),
                                round(self.analysis['precision_0'], 2),
                                round(self.analysis['recall_0'], 2),
                                round(self.analysis['f1_score_0'], 2),

                                round(self.analysis['pred_pos_rate_1'], 2),
                                round(self.analysis['accuracy_1'], 2),
                                round(self.analysis['precision_1'], 2),
                                round(self.analysis['recall_1'], 2),
                                round(self.analysis['f1_score_1'], 2),

                                round(self.analysis['pred_pos_rate_2'], 2),
                                round(self.analysis['accuracy_2'], 2),
                                round(self.analysis['precision_2'], 2),
                                round(self.analysis['recall_2'], 2),
                                round(self.analysis['f1_score_2'], 2),

                                round(self.analysis['pred_pos_rate_3'], 2),
                                round(self.analysis['accuracy_3'], 2),
                                round(self.analysis['precision_3'], 2),
                                round(self.analysis['recall_3'], 2),
                                round(self.analysis['f1_score_3'], 2)
                                ]
                dump_analysis = toolbox.add_row_to_df(dump_analysis, current_analysis)
  
        if debug:
            dump_predictions.to_csv('./tmp/cross_validation_results.csv')
            dump_analysis.to_csv('./tmp/cross_validation_analysis.csv')

        results["average_accuracy"] = sum(results["accuracies"]) / len(results["accuracies"])
        return results

'''
LSTM1

Input:
- epochs = 170
- seq_len = 21
- lstm_size = 100
- batch_size = 10
'''        
class ClassifierLSTM1(ClassifierLSTM):
    def __init__(self, dataframe, data_splitter, params = None):
        super().__init__(dataframe, data_splitter, params)

        self.lstm_size = 100
        if params:
            self.lstm_size = params.get("lstm_size", self.lstm_size)
    
    def build(self):
        print("[Build ClassifierLSTM1]")
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_size, input_shape=(1, self.input_size)))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

'''
LSTM2

Input:
- epochs = 170
- seq_len = 21
- lstm_size = 100
- batch_size = 10
'''  
class ClassifierLSTM2(ClassifierLSTM):
    def __init__(self, dataframe, data_splitter, params = None):
        super().__init__(dataframe, data_splitter, params)
    
        self.lstm_size = 100
        if params:
            self.lstm_size = params.get("lstm_size", self.lstm_size)

    def build(self):
        print("[Build ClassifierLSTM2]")
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_size, input_shape=(1, self.input_size)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.lstm_size, activation='sigmoid'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

'''
LSTM3
ref : Functional API : https://keras.io/guides/functional_api/

Input:
- epochs = 170
- seq_len = 21
- lstm_size = 100
- batch_size = 10
'''  
class ClassifierLSTM3(ClassifierLSTM):
    def __init__(self, dataframe, data_splitter, params = None):
        super().__init__(dataframe, data_splitter, params)
    
        self.lstm_size = 100
        if params:
            self.lstm_size = params.get("lstm_size", self.lstm_size)
    
    def build(self):
        print("[Build ClassifierLSTM3]")

        inputs = keras.Input(shape=(1, self.input_size))
        x = layers.LSTM(self.lstm_size, activation="sigmoid")(inputs)
        outputs = layers.Dense(1)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

'''
LSTMHao2020
ref : 
"Predicting the Trend of Stock Market Index Using the Hybrid Neural Network Based on Multiple Time Scale Feature Learning", Yaping Hao - Qiang Gao, 2020
https://www.mdpi.com/2076-3417/10/11/3961
https://www.researchgate.net/publication/342045600_Predicting_the_Trend_of_Stock_Market_Index_Using_the_Hybrid_Neural_Network_Based_on_Multiple_Time_Scale_Feature_Learning

Input:
- epochs = 170
- seq_len = 21
- batch_size = 10
'''
class ClassifierLSTMHao2020(ClassifierLSTM):
    def __init__(self, dataframe, data_splitter, params = None):
        super().__init__(dataframe, data_splitter, params)
    
    def build(self):
        print("[Build ClassifierLSTMHao2020]")

        inputs = keras.Input(shape=(1, self.input_size), name="Input")
        convmax2 = Conv1D(10, 3, activation="relu", padding='same')(inputs)
        convmax2 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(convmax2)
        
        convmax3 = Conv1D(20, 3, activation="relu", padding='same')(convmax2)
        convmax3 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(convmax3)
        F3 = layers.LSTM(10, activation="sigmoid")(convmax3)

        F2 = layers.LSTM(10, activation="sigmoid")(convmax2)
        
        F1 = layers.LSTM(10, activation="sigmoid")(inputs)

        concat = layers.concatenate([F1, F2, F3])
 
        outputs = layers.Dense(10)(concat)
        outputs = layers.Dense(1, name="Output")(outputs)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
        backend.set_value(self.model.optimizer.learning_rate, 0.001)

'''
BiLSTM
ref : https://towardsdatascience.com/the-beginning-of-a-deep-learning-trading-bot-part1-95-accuracy-is-not-enough-c338abc98fc2

Input:
- epochs = 170
- seq_len = 21
- batch_size = 10
'''
class ClassifierBiLSTM(ClassifierLSTM):
    def __init__(self, dataframe, data_splitter, params = None):
        super().__init__(dataframe, data_splitter, params)
    
    def build(self):
        print("[Build ClassifierBiLSTM]")

        in_seq = keras.Input(shape=(1, self.input_size))
      
        x = layers.Bidirectional(LSTM(self.seq_len, return_sequences=True))(in_seq)
        x = layers.Bidirectional(LSTM(self.seq_len, return_sequences=True))(x)
        x = layers.Bidirectional(LSTM(int(self.seq_len/2), return_sequences=True))(x) 
      
        avg_pool = layers.GlobalAveragePooling1D()(x)
        max_pool = layers.GlobalMaxPooling1D()(x)
        conc = layers.concatenate([avg_pool, max_pool])
        conc = layers.Dense(int(self.seq_len/2), activation="relu")(conc)
        out = layers.Dense(1, activation="sigmoid")(conc)      

        self.model = keras.Model(inputs=in_seq, outputs=out)
        self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])    

'''
CNN + Bi-LSTM model
ref : https://towardsdatascience.com/the-beginning-of-a-deep-learning-trading-bot-part1-95-accuracy-is-not-enough-c338abc98fc2

Input:
- epochs = 170
- seq_len = 21
- batch_size = 10
'''
class ClassifierCNNBiLSTM(ClassifierLSTM):
    def __init__(self, dataframe, data_splitter, params = None):
        super().__init__(dataframe, data_splitter, params)
    
    def inception_a(self, layer_in, c7):
        branch1x1_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
        branch1x1 = BatchNormalization()(branch1x1_1)
        branch1x1 = ReLU()(branch1x1)

        branch5x5_1 = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(layer_in)
        branch5x5 = BatchNormalization()(branch5x5_1)
        branch5x5 = ReLU()(branch5x5)
        branch5x5 = Conv1D(c7, kernel_size=5, padding='same', use_bias=False)(branch5x5)
        branch5x5 = BatchNormalization()(branch5x5)
        branch5x5 = ReLU()(branch5x5)  

        branch3x3_1 = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(layer_in)
        branch3x3 = BatchNormalization()(branch3x3_1)
        branch3x3 = ReLU()(branch3x3)
        branch3x3 = Conv1D(c7, kernel_size=3, padding='same', use_bias=False)(branch3x3)
        branch3x3 = BatchNormalization()(branch3x3)
        branch3x3 = ReLU()(branch3x3)
        branch3x3 = Conv1D(c7, kernel_size=3, padding='same', use_bias=False)(branch3x3)
        branch3x3 = BatchNormalization()(branch3x3)
        branch3x3 = ReLU()(branch3x3) 

        branch_pool = AveragePooling1D(pool_size=(3), strides=1, padding='same')(layer_in)
        branch_pool = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(branch_pool)
        branch_pool = BatchNormalization()(branch_pool)
        branch_pool = ReLU()(branch_pool)
        outputs = Concatenate(axis=-1)([branch1x1, branch5x5, branch3x3, branch_pool])
        return outputs


    def inception_b(self, layer_in, c7):
        branch3x3 = Conv1D(c7, kernel_size=3, padding="same", strides=2, use_bias=False)(layer_in)
        branch3x3 = BatchNormalization()(branch3x3)
        branch3x3 = ReLU()(branch3x3)  

        branch3x3dbl = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
        branch3x3dbl = BatchNormalization()(branch3x3dbl)
        branch3x3dbl = ReLU()(branch3x3dbl)  
        branch3x3dbl = Conv1D(c7, kernel_size=3, padding="same", use_bias=False)(branch3x3dbl)  
        branch3x3dbl = BatchNormalization()(branch3x3dbl)
        branch3x3dbl = ReLU()(branch3x3dbl)  
        branch3x3dbl = Conv1D(c7, kernel_size=3, padding="same", strides=2, use_bias=False)(branch3x3dbl)    
        branch3x3dbl = BatchNormalization()(branch3x3dbl)
        branch3x3dbl = ReLU()(branch3x3dbl)   

        branch_pool = MaxPooling1D(pool_size=3, strides=2, padding="same")(layer_in)

        outputs = Concatenate(axis=-1)([branch3x3, branch3x3dbl, branch_pool])
        return outputs


    def inception_c(self, layer_in, c7):
        branch1x1_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
        branch1x1 = BatchNormalization()(branch1x1_1)
        branch1x1 = ReLU()(branch1x1)   

        branch7x7_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
        branch7x7 = BatchNormalization()(branch7x7_1)
        branch7x7 = ReLU()(branch7x7)   
        branch7x7 = Conv1D(c7, kernel_size=(7), padding="same", use_bias=False)(branch7x7)
        branch7x7 = BatchNormalization()(branch7x7)
        branch7x7 = ReLU()(branch7x7)  
        branch7x7 = Conv1D(c7, kernel_size=(1), padding="same", use_bias=False)(branch7x7)  
        branch7x7 = BatchNormalization()(branch7x7)
        branch7x7 = ReLU()(branch7x7)   

        branch7x7dbl_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)  
        branch7x7dbl = BatchNormalization()(branch7x7dbl_1)
        branch7x7dbl = ReLU()(branch7x7dbl)  
        branch7x7dbl = Conv1D(c7, kernel_size=(7), padding="same", use_bias=False)(branch7x7dbl)  
        branch7x7dbl = BatchNormalization()(branch7x7dbl)
        branch7x7dbl = ReLU()(branch7x7dbl) 
        branch7x7dbl = Conv1D(c7, kernel_size=(1), padding="same", use_bias=False)(branch7x7dbl)  
        branch7x7dbl = BatchNormalization()(branch7x7dbl)
        branch7x7dbl = ReLU()(branch7x7dbl)  
        branch7x7dbl = Conv1D(c7, kernel_size=(7), padding="same", use_bias=False)(branch7x7dbl)  
        branch7x7dbl = BatchNormalization()(branch7x7dbl)
        branch7x7dbl = ReLU()(branch7x7dbl)  
        branch7x7dbl = Conv1D(c7, kernel_size=(1), padding="same", use_bias=False)(branch7x7dbl)  
        branch7x7dbl = BatchNormalization()(branch7x7dbl)
        branch7x7dbl = ReLU()(branch7x7dbl)  

        branch_pool = AveragePooling1D(pool_size=3, strides=1, padding='same')(layer_in)
        branch_pool = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(branch_pool)
        branch_pool = BatchNormalization()(branch_pool)
        branch_pool = ReLU()(branch_pool)  

        outputs = Concatenate(axis=-1)([branch1x1, branch7x7, branch7x7dbl, branch_pool])
        return outputs

    def build(self):
        print("[Build ClassifierCNNBiLSTM]")

        in_seq = keras.Input(shape=(1, self.input_size))
        c7 = int(self.seq_len/4)

        x = self.inception_a(in_seq, c7)
        x = self.inception_a(x, c7)
        x = self.inception_b(x, c7)
        x = self.inception_b(x, c7)
        x = self.inception_c(x, c7)
        x = self.inception_c(x, c7)    
            
        x = layers.Bidirectional(LSTM(self.seq_len, return_sequences=True))(x)
        x = layers.Bidirectional(LSTM(self.seq_len, return_sequences=True))(x)
        x = layers.Bidirectional(LSTM(int(self.seq_len/2), return_sequences=True))(x) 
            
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        conc = Dense(64, activation="relu")(conc)
        out = Dense(1, activation="sigmoid")(conc)      

        self.model = Model(inputs=in_seq, outputs=out)
        self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])     
