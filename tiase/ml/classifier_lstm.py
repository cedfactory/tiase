from . import toolbox,analysis,classifier
from scikeras.wrappers import KerasClassifier
import xml.etree.cElementTree as ET

from rich import print,inspect

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Reshape, Dense, LSTM, Dropout, ReLU, BatchNormalization, AveragePooling1D, Conv1D, concatenate, Concatenate, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
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
# https://github.com/MohammadFneish7/Keras_LSTM_Diagram
# https://www.adriangb.com/scikeras/stable/migration.html
#
class ClassifierLSTM(classifier.Classifier):
    def __init__(self, data_splitter, params = None):
        super().__init__(data_splitter, params)

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

    def _transform_X(self, x):
        return np.reshape(x, (x.shape[0], 1, x.shape[1]))

    def fit(self):
        # create the model
        tf.random.set_seed(20)
        np.random.seed(10)

        self.build()

        #print(self.model.summary())
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def get_model(self):
        return self.model

    def get_param_grid(self):
        return self.param_grid

    def get_analysis(self):
        self.y_test_prob = self.model.predict(self.X_test)

        self.threshold, self.y_test_pred = toolbox.get_classification_threshold("naive", self.y_test, self.y_test_prob)

        self.analysis = analysis.classification_analysis(self.X_test, self.y_test, self.y_test_pred, self.y_test_prob)
        self.analysis["history"] = self.model.history_
        return self.analysis

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename+".hdf5")

    def save(self, filename):
        model_analysis = self.get_analysis()

        model_filename = filename+".hdf5"
        self.model.model_.save(model_filename)
        
        x_normalizer_filename = filename+"_x_normalizer.gz"
        toolbox.serialize(self.data_splitter.normalizer, x_normalizer_filename)

        analysis.export_history(filename, self.analysis["history"])
        analysis.export_roc_curve(model_analysis["y_test"], model_analysis["y_test_prob"], filename+"_classification_roc_curve.png")
        analysis.export_confusion_matrix(model_analysis["confusion_matrix"], filename+"_classification_confusion_matrix.png")

        root = ET.Element("model")
        ET.SubElement(root, "file").text = model_filename
        ET.SubElement(root, "x_normaliser").text = x_normalizer_filename
        ET.SubElement(root, "accuracy").text = "{:.2f}".format(model_analysis["accuracy"])
        ET.SubElement(root, "precision").text = "{:.2f}".format(model_analysis["precision"])
        ET.SubElement(root, "recall").text = "{:.2f}".format(model_analysis["recall"])
        ET.SubElement(root, "f1_score").text = "{:.2f}".format(model_analysis["f1_score"])
        tree = ET.ElementTree(root)

        xmlfilename = filename+'.xml'
        tree.write(xmlfilename)


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

            # create the model
            tf.random.set_seed(20)
            np.random.seed(10)

            self.X_train = self._transform_X(self.X_train)
            self.X_test = self._transform_X(self.X_test)

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
    def __init__(self, data_splitter, params = None):
        super().__init__(data_splitter, params)

        self.lstm_size = 100
        self.param_grid = {
            'epochs': [10, 20, 30, 40],
            'batch_size': [5, 10, 15, 20, 25]
        }
        if params:
            self.lstm_size = params.get("lstm_size", self.lstm_size)
            self.param_grid = params.get("param_grid", self.param_grid)

    def build(self):
        print("[Build ClassifierLSTM1]")
        def create_keras_classifier():
            model = Sequential()
            model.add(Reshape((1, self.input_size), input_shape=(self.input_size,)))
            model.add(LSTM(self.lstm_size, input_shape=(1, self.input_size)))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        self.model = KerasClassifier(model=create_keras_classifier)

'''
LSTM2

Input:
- epochs = 100
- seq_len = 21
- lstm_size = 100
- batch_size = 10
'''  
class ClassifierLSTM2(ClassifierLSTM):
    def __init__(self, data_splitter, params = None):
        super().__init__(data_splitter, params)
    
        self.lstm_size = 100
        if params:
            self.lstm_size = params.get("lstm_size", self.lstm_size)

    def build(self):
        print("[Build ClassifierLSTM2]")
        def create_keras_classifier():
            model = Sequential()
            model.add(Reshape((1, self.input_size), input_shape=(self.input_size,)))
            model.add(LSTM(self.lstm_size, input_shape=(1, self.input_size)))
            model.add(Dropout(0.5))
            model.add(Dense(self.lstm_size, activation='sigmoid'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        self.model = KerasClassifier(model=create_keras_classifier)


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
    def __init__(self, data_splitter, params = None):
        super().__init__(data_splitter, params)
    
        self.lstm_size = 100
        if params:
            self.lstm_size = params.get("lstm_size", self.lstm_size)
    
    def build(self):
        print("[Build ClassifierLSTM3]")
        def create_keras_classifier():
            inputs = keras.Input(shape=(self.input_size,))
            reshaped = Reshape((1, self.input_size), input_shape=(self.input_size,))(inputs)
            x = layers.LSTM(self.lstm_size, activation="sigmoid")(reshaped)
            outputs = layers.Dense(1)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
            return model

        self.model = KerasClassifier(model=create_keras_classifier)

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
    def __init__(self, data_splitter, params = None):
        super().__init__(data_splitter, params)
    
    def build(self):
        print("[Build ClassifierLSTMHao2020]")
        def create_keras_classifier():
            inputs = keras.Input(shape=(self.input_size,), name="Input")
            reshaped = Reshape((1, self.input_size), input_shape=(self.input_size,))(inputs)
            convmax2 = Conv1D(10, 3, activation="relu", padding='same')(reshaped)
            convmax2 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(convmax2)
            
            convmax3 = Conv1D(20, 3, activation="relu", padding='same')(convmax2)
            convmax3 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(convmax3)
            F3 = layers.LSTM(10, activation="sigmoid")(convmax3)

            F2 = layers.LSTM(10, activation="sigmoid")(convmax2)
            
            F1 = layers.LSTM(10, activation="sigmoid")(reshaped)

            concat = layers.concatenate([F1, F2, F3])

            outputs = layers.Dense(10)(concat)
            outputs = layers.Dense(1, name="Output")(outputs)

            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
            backend.set_value(model.optimizer.learning_rate, 0.001)

            return model

        self.model = KerasClassifier(model=create_keras_classifier)

'''
BiLSTM
ref : https://towardsdatascience.com/the-beginning-of-a-deep-learning-trading-bot-part1-95-accuracy-is-not-enough-c338abc98fc2

Input:
- epochs = 170
- seq_len = 21
- batch_size = 10
'''
class ClassifierBiLSTM(ClassifierLSTM):
    def __init__(self, data_splitter, params = None):
        super().__init__(data_splitter, params)
    
    def build(self):
        print("[Build ClassifierBiLSTM]")
        def create_keras_classifier():
            in_seq = keras.Input(shape=(self.input_size,))
            reshaped = Reshape((1, self.input_size), input_shape=(self.input_size,))(in_seq)
            
            x = layers.Bidirectional(LSTM(self.seq_len, return_sequences=True))(reshaped)
            x = layers.Bidirectional(LSTM(self.seq_len, return_sequences=True))(x)
            x = layers.Bidirectional(LSTM(int(self.seq_len/2), return_sequences=True))(x) 
        
            avg_pool = layers.GlobalAveragePooling1D()(x)
            max_pool = layers.GlobalMaxPooling1D()(x)
            conc = layers.concatenate([avg_pool, max_pool])
            conc = layers.Dense(int(self.seq_len/2), activation="relu")(conc)
            out = layers.Dense(1, activation="sigmoid")(conc)      

            model = keras.Model(inputs=in_seq, outputs=out)
            model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

            return model

        self.model = KerasClassifier(model=create_keras_classifier)

'''
CNN + Bi-LSTM model
ref : https://towardsdatascience.com/the-beginning-of-a-deep-learning-trading-bot-part1-95-accuracy-is-not-enough-c338abc98fc2

Input:
- epochs = 170
- seq_len = 21
- batch_size = 10
'''
class ClassifierCNNBiLSTM(ClassifierLSTM):
    def __init__(self, data_splitter, params = None):
        super().__init__(data_splitter, params)
    
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
        def create_keras_classifier():
            in_seq = keras.Input(shape=(self.input_size,))
            reshaped = Reshape((1, self.input_size), input_shape=(self.input_size,))(in_seq)
            c7 = int(self.seq_len/4)

            x = self.inception_a(reshaped, c7)
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

            model = Model(inputs=in_seq, outputs=out)
            model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

            return model

        self.model = KerasClassifier(model=create_keras_classifier)