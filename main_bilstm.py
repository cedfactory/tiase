# https://towardsdatascience.com/the-beginning-of-a-deep-learning-trading-bot-part1-95-accuracy-is-not-enough-c338abc98fc2

from tiase.fimport import visu
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Bidirectional,LSTM,Dense,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
print('Tensorflow version: {}'.format(tf.__version__))

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')

import xml.etree.cElementTree as ET

#
# Read Data
#
def read_data(filename):
    df = pd.read_csv(filename, delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # Replace 0 to avoid dividing by 0 later on
    df['Volume'].replace(to_replace=0, method='ffill', inplace=True) 
    df.sort_values('Date', inplace=True)

    return df

#
# plot_loss found in history
# todo : to move into mltoolbox
#
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    ymax = max(history.history['loss']) / 2
    plt.ylim([0, ymax])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

class BiLSTM:

    def __init__(self):
        self.seq_len = 64 #128

    def import_data(self, df):
        self.df = df
        self.normalize_data()


    def normalize_data(self):
        '''Calculate percentage change'''
        
        self.df['Open'] = self.df['Open'].pct_change() # Create arithmetic returns column
        self.df['High'] = self.df['High'].pct_change() # Create arithmetic returns column
        self.df['Low'] = self.df['Low'].pct_change() # Create arithmetic returns column
        self.df['Close'] = self.df['Close'].pct_change() # Create arithmetic returns column
        self.df['Volume'] = self.df['Volume'].pct_change()

        self.df.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values

        ###############################################################################
        '''Normalize price columns'''

        self.min_return = min(self.df[['Open', 'High', 'Low', 'Close']].min(axis=0))
        self.max_return = max(self.df[['Open', 'High', 'Low', 'Close']].max(axis=0))
        print("min_return = {}    max_return = {}".format(self.min_return, self.max_return))

        # Min-max normalize price columns (0-1 range)
        self.df['Open'] = (self.df['Open'] - self.min_return) / (self.max_return - self.min_return)
        self.df['High'] = (self.df['High'] - self.min_return) / (self.max_return - self.min_return)
        self.df['Low'] = (self.df['Low'] - self.min_return) / (self.max_return - self.min_return)
        self.df['Close'] = (self.df['Close'] - self.min_return) / (self.max_return - self.min_return)

        ###############################################################################
        '''Normalize volume column'''

        self.min_volume = self.df['Volume'].min(axis=0)
        self.max_volume = self.df['Volume'].max(axis=0)
        print("min_volume = {}    max_volume = {}".format(self.min_volume, self.max_volume))

        # Min-max normalize volume columns (0-1 range)
        self.df['Volume'] = (self.df['Volume'] - self.min_volume) / (self.max_volume - self.min_volume)

    def create_training_validation_test_split(self):
        ###############################################################################
        '''Create training, validation and test split'''

        times = sorted(self.df.index.values)
        self.last_10pct = sorted(self.df.index.values)[-int(0.1*len(times))] # Last 10% of series
        self.last_20pct = sorted(self.df.index.values)[-int(0.2*len(times))] # Last 20% of series

        self.df_train = self.df[(self.df.index < self.last_20pct)]  # Training data are 80% of total data
        self.df_val = self.df[(self.df.index >= self.last_20pct) & (self.df.index < self.last_10pct)]
        self.df_test = self.df[(self.df.index >= self.last_10pct)]

        # Remove date column
        self.df_train.drop(columns=['Date'], inplace=True)
        self.df_val.drop(columns=['Date'], inplace=True)
        self.df_test.drop(columns=['Date'], inplace=True)

        # Convert pandas columns into arrays
        self.train_data = self.df_train.values
        self.val_data = self.df_val.values
        self.test_data = self.df_test.values
        print('Training data shape: {}'.format(self.train_data.shape))
        print('Validation data shape: {}'.format(self.val_data.shape))
        print('Test data shape: {}'.format(self.test_data.shape))

    def plot_daily_changes(self):
        fig = plt.figure(figsize=(15,10))
        st = fig.suptitle("Data Separation", fontsize=20)
        st.set_y(0.92)

        ###############################################################################

        ax1 = fig.add_subplot(211)
        ax1.plot(np.arange(self.train_data.shape[0]), self.df_train['Close'], label='Training data')

        ax1.plot(np.arange(self.train_data.shape[0], 
                        self.train_data.shape[0]+self.val_data.shape[0]), self.df_val['Close'], label='Validation data')

        ax1.plot(np.arange(self.train_data.shape[0]+self.val_data.shape[0], 
                        self.train_data.shape[0]+self.val_data.shape[0]+self.test_data.shape[0]), self.df_test['Close'], label='Test data')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normalized Closing Returns')

        ###############################################################################

        ax2 = fig.add_subplot(212)
        ax2.plot(np.arange(self.train_data.shape[0]), self.df_train['Volume'], label='Training data')

        ax2.plot(np.arange(self.train_data.shape[0], 
                        self.train_data.shape[0]+self.val_data.shape[0]), self.df_val['Volume'], label='Validation data')

        ax2.plot(np.arange(self.train_data.shape[0]+self.val_data.shape[0], 
                        self.train_data.shape[0]+self.val_data.shape[0]+self.test_data.shape[0]), self.df_test['Volume'], label='Test data')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Normalized Volume Changes')

        plt.legend(loc='best')

    def create_training_validation_test_data(self):
        # Training data
        self.X_train, self.y_train = [], []
        for i in range(self.seq_len, len(self.train_data)):
            self.X_train.append(self.train_data[i-self.seq_len:i]) # Chunks of training data with a length of seq_len df-rows
            self.y_train.append(self.train_data[:, 3][i]) #Value of 4th column (Close Price) of df-row seq_len+1
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)

        ###############################################################################

        # Validation data
        self.X_val, self.y_val = [], []
        for i in range(self.seq_len, len(self.val_data)):
            self.X_val.append(self.val_data[i-self.seq_len:i])
            self.y_val.append(self.val_data[:, 3][i])
        self.X_val, self.y_val = np.array(self.X_val), np.array(self.y_val)

        ###############################################################################

        # Test data
        self.X_test, self.y_test = [], []
        for i in range(self.seq_len, len(self.test_data)):
            self.X_test.append(self.test_data[i-self.seq_len:i])
            self.y_test.append(self.test_data[:, 3][i])    
        self.X_test, self.y_test = np.array(self.X_test), np.array(self.y_test)

        print(self.X_train.shape, self.y_train.shape)
        print(self.X_val.shape, self.y_val.shape)

    #
    # Bi-LSTM model
    #
    def create_model1(self):
        in_seq = Input(shape = (self.seq_len, 5))
      
        x = Bidirectional(LSTM(self.seq_len, return_sequences=True))(in_seq)
        x = Bidirectional(LSTM(self.seq_len, return_sequences=True))(x)
        x = Bidirectional(LSTM(int(self.seq_len/2), return_sequences=True))(x) 
      
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        conc = Dense(int(self.seq_len/2), activation="relu")(conc)
        out = Dense(1, activation="linear")(conc)      

        self.model = Model(inputs=in_seq, outputs=out)
        self.model.compile(loss="mse", optimizer="adam", metrics=['mse', 'mae', 'mape'])    
        #self.model.summary()


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


    #
    # CNN + Bi-LSTM model
    #
    def create_model2(self):
        in_seq = Input(shape=(self.seq_len, 5))

        c7 = int(self.seq_len/4)
        x = self.inception_a(in_seq, c7)
        x = self.inception_a(x, c7)
        x = self.inception_b(x, c7)
        x = self.inception_b(x, c7)
        x = self.inception_c(x, c7)
        x = self.inception_c(x, c7)    
            
        x = Bidirectional(LSTM(self.seq_len, return_sequences=True))(x)
        x = Bidirectional(LSTM(self.seq_len, return_sequences=True))(x)
        x = Bidirectional(LSTM(int(self.seq_len/2), return_sequences=True))(x) 
            
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        conc = Dense(64, activation="relu")(conc)
        out = Dense(1, activation="sigmoid")(conc)      

        self.model = Model(inputs=in_seq, outputs=out)
        self.model.compile(loss="mse", optimizer="adam", metrics=['mae', 'mape'])     
 


    def fit_model(self, epochs):
        #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        callback = tf.keras.callbacks.ModelCheckpoint('Bi-LSTM.hdf5', monitor='val_loss', save_best_only=True, verbose=1)

        history = self.model.fit(self.X_train, self.y_train,
                    batch_size=2048,
                    verbose=2,
                    callbacks=[callback],
                    epochs=epochs,
                    validation_data=(self.X_val, self.y_val),)    

        return history

    def train_model(self, epochs):
        self.create_training_validation_test_split()
        #this.df_train.head()

        #this.PlotDailyChanges()
        self.create_training_validation_test_data()
        self.create_model1()
        self.history = self.fit_model(epochs)

    def display_stats(self):
        plot_loss(self.history)
        self.evaluate_predictions()


    def evaluate_predictions(self):
        '''Evaluate predictions and metrics'''

        #Calculate predication for training, validation and test data
        train_pred = self.model.predict(self.X_train)
        val_pred = self.model.predict(self.X_val)
        test_pred = self.model.predict(self.X_test)

        #Print evaluation metrics for all datasets
        train_eval = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        val_eval = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        test_eval = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(' ')
        print('Evaluation metrics')
        print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
        print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
        print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))
        self.test_mse = test_eval[0]
        self.test_mae = test_eval[1]
        self.test_mape = test_eval[2]

        '''Display results'''

        fig = plt.figure(figsize=(15,15))
        st = fig.suptitle("Bi-LSTM Model", fontsize=22)
        st.set_y(1.02)

        #Plot training data results
        ax11 = fig.add_subplot(311)
        ax11.plot(self.train_data[:, 3], label='Closing Returns')
        ax11.plot(train_pred, color='yellow', linewidth=3, label='Predicted Closing Returns')
        ax11.set_title("Training Data", fontsize=18)
        ax11.set_xlabel('Date')
        ax11.set_ylabel('Closing Returns')

        #Plot validation data results
        ax21 = fig.add_subplot(312)
        ax21.plot(self.val_data[:, 3], label='Closing Returns')
        ax21.plot(val_pred, color='yellow', linewidth=3, label='Predicted Closing Returns')
        ax21.set_title("Validation Data", fontsize=18)
        ax21.set_xlabel('Date')
        ax21.set_ylabel('Closing Returns')

        #Plot test data results
        ax31 = fig.add_subplot(313)
        ax31.plot(self.test_data[:, 3], label='Closing Returns')
        ax31.plot(test_pred, color='yellow', linewidth=3, label='Predicted Closing Returns')
        ax31.set_title("Test Data", fontsize=18)
        ax31.set_xlabel('Date')
        ax31.set_ylabel('Closing Returns')

        plt.tight_layout()
        plt.legend(loc='best')

    def save_model(self, name):
        filename = './'+name+'.hdf5'
        print(filename)
        print(self.min_return)
        self.model.save(filename)

        root = ET.Element("model")
        ET.SubElement(root, "file").text = name+'.hdf5'
        ET.SubElement(root, "min_return").text = str(self.min_return)
        ET.SubElement(root, "max_return").text = str(self.max_return)
        ET.SubElement(root, "min_volume").text = str(self.min_volume)
        ET.SubElement(root, "max_volume").text = str(self.max_volume)
        ET.SubElement(root, "mae").text = str(self.test_mse)
        ET.SubElement(root, "mse").text = str(self.test_mae)
        ET.SubElement(root, "mape").text = str(self.test_mape)

        xmlfilename = './'+name+'.xml'

        tree = ET.ElementTree(root)
        print(xmlfilename)
        tree.write(xmlfilename)

    def load_model(self, filename):

        tree = ET.parse(filename)
        root = tree.getroot()
        
        self.model = tf.keras.models.load_model(root[0].text)
        self.min_return = float(root[1].text)
        self.max_return = float(root[2].text)
        self.min_volume = float(root[3].text)
        self.max_volume = float(root[4].text)
        print("Load model from {} : {}".format(filename, root[0].text))
        print("min_return = {}    max_return = {}".format(self.min_return, self.max_return))
        print("min_volume = {}    max_volume = {}".format(self.min_volume, self.max_volume))

    def denormalize_value(self, normalized_value, min_value, max_value):
        return normalized_value * (max_value - min_value) + min_value

    def denormalize_values(self, normalized_values, min_value, max_value):
        return [self.denormalize_value(normalize_value, min_value, max_value) for normalize_value in normalized_values]

    def stats_for_trends(self):
        df2 = self.df.copy()
        df2.drop(columns=['Date'], inplace=True)
        #print(df2[0:67])
        data = df2.values
        sequence = []
        expected = []
        lastClose = []
        n = len(df2.index)-self.seq_len-2
        for index in range(n):
            sequence.append(data[index:index+self.seq_len])
            lastClose.append(data[:, 3][index+self.seq_len-1])
            expected.append(data[:, 3][index+self.seq_len])
        sequence = np.array(sequence)
        predictions = self.model.predict(sequence)
        print("stats_for_trends")
        denormalized_expectations = self.denormalize_values(expected, self.min_return, self.max_return)
        print(denormalized_expectations)
        denormalized_expectations = self.denormalize_values(predictions, self.min_return, self.max_return)
        print(denormalized_expectations)

        all_ok = 0
        down = 0
        up = 0
        for index in range(n):
            predicted = denormalized_expectations[index][0]
            if denormalized_expectations[index] > 0:
                up = up + 1
            else:
                down = down + 1
            if (denormalized_expectations[index]*predicted) > 0:
                ok = 1
                all_ok = all_ok + 1
            else:
                ok = 0
            #print("{}   {} {} vs {}".format(ok, lastClose[index], expected[index], predicted))
        print("StatsForTrends : {} down / {} up".format(down, up))
        print("StatsForTrends : {} / {} => {:.2f}%".format(all_ok, n, 100 * all_ok / n))
    
    def make_trend_prediction_for_next_tick(self):
        df2 = self.df.copy()
        df2.drop(columns=['Date'], inplace=True)
        #print(df2[0:67])
        data = df2.values
        sequence = []
        expected = []
        last_close = []
        n = len(df2.index)
        index = n-self.seq_len

        sequence.append(data[index:index+self.seq_len])
        last_close = data[:, 3][index+self.seq_len-1]
        print(last_close)
        sequence = np.array(sequence)
        normalized_prediction = self.model.predict(sequence)
        prediction = self.denormalize_value(normalized_prediction[0][0], self.min_return, self.max_return)
        return prediction





prices = './tiase/data/IBM_Prices.csv'
value = "IBM"
df = read_data(prices)

visu.display_from_dataframe(df, 'Close', 'close.png')
#df.head()

bilstm = BiLSTM()
bilstm.import_data(df)
bilstm.train_model(epochs = 1)
bilstm.display_stats()
bilstm.save_model('bi_lstm_'+value)

xmlfile = "./bi_lstm_"+value+".xml"
df = read_data(prices)

#importers.display_from_dataframe(df, 'Close')
#importers.display_from_dataframe(df, 'Volume')
#df.head()

bilstm = BiLSTM()
bilstm.import_data(df)
bilstm.load_model(xmlfile)
bilstm.stats_for_trends()
prediction = bilstm.make_trend_prediction_for_next_tick()
print(prediction)
