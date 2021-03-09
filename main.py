from importlib import reload
#from lib.importers import importers
#reload(importers)
from lib.ml import ml_bilstm
reload(ml_bilstm)
from lib.ml import ml
reload(ml)

import pandas as pd
import datetime
import matplotlib.pyplot as plt

import numpy as np


prices = './lib/data/CAC40/_AI.PA.csv'
value = "AI.PA"
df = ml_bilstm.ReadData(prices)

#importers.DisplayFromDataframe(df, 'Close')
#importers.DisplayFromDataframe(df, 'Volume')
#df.head()

bilstm = ml_bilstm.BiLSTM(df)


bilstm.NormalizeData()
bilstm.CreateTrainingValidationTestSplit()
bilstm.df_train.head()

#bilstm.PlotDailyChanges()
bilstm.CreateTrainingValidationTestData()
bilstm.CreateModel1()
history = bilstm.FitModel(epochs = 10)
ml.PlotLoss(history)
bilstm.EvaluatePredictions()
bilstm.SaveModel('bi_lstm_'+value)

xmlfile = "./bi_lstm_ACA.PA.xml"
df = ml_bilstm.ReadData(prices)

#importers.DisplayFromDataframe(df, 'Close')
#importers.DisplayFromDataframe(df, 'Volume')
#df.head()

bilstm = ml_bilstm.BiLSTM(df)
bilstm.NormalizeData()
bilstm.LoadModel(xmlfile)
bilstm.StatsForTrends()
[prediction, trend] = bilstm.MakeTrendPredictionForNextTick()
print(prediction)
print(trend)
#[expected, predicted] = bilstm.MakePrediction(0, 3000)
