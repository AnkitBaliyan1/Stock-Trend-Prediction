import joblib
import pickle
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import tensorflow

print("start")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2023-01-01'


user = 'AAPL'
df = yf.download(user,start,end)
print(df.head())


plt.plot(df.Close)
# plt.show()

# importing model
print("importing model now")
model = joblib.load("keras_model.pkl")

print(model)


