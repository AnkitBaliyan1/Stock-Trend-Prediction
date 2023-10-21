import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st



start = '2010-01-01'
end = '2023-01-01'


st.title("Stock Trend Prediction")

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input,start,end)

# Describing data

st.subheader('Data from 2010 - 2022')
