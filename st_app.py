
import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib



start = '2010-01-01'
end = '2023-01-01'


st.title("Stock Trend Prediction")

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input,start,end)

# Describing data

st.subheader('Data from 2010 - 2022')
st.write(df.describe())

# Visualisation
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with MA100')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
plt.plot(ma100,'g')
st.pyplot(fig)




st.subheader('Closing Price vs Time Chart with MA100 and MA200')
ma200 = df.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
plt.plot(ma100,'g')
plt.plot(ma200,'r')
st.pyplot(fig2)






data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])



scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)




x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train),np.array(y_train)





# load my model
model = joblib.load('keras_model.pkl')





# testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)


input_data=scaler.fit_transform(final_df)



x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test=np.array(x_test)
y_test=np.array(y_test)

y_predicted = model.predict(x_test)



scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



# final graph

st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)