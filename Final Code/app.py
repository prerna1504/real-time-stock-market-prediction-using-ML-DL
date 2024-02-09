import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.models import load_model
import streamlit as st
import pandas_datareader as reader
from sklearn.svm import SVR
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,precision_score
import matplotlib.pyplot as plt


st.set_page_config(page_title="Stock Market Price Prediction App")
st.title('Predict Stock Market Prices')

# Get user input for the ticker symbol
ticker = st.text_input("Enter the ticker symbol of the company you want to predict: ")
data = yf.download(ticker)

#Get Company Profile Details of the user entered ticker symbol
st.subheader('Company Details')
def get_company_profile(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info
company_profile = get_company_profile(ticker)
st.write(f"Company Name: {company_profile['longName']}")
st.write(f"Industry: {company_profile['industry']}")
st.write(f"Sector: {company_profile['sector']}")
st.write(f"Country: {company_profile['country']}")

#Get Company description of the user entered ticker symbol
def get_company_description(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info['longBusinessSummary']
st.subheader('Description')
company_description = get_company_description(ticker)
st.write(f"{company_description}")

#Describing Data
st.subheader('Data fetched from Yahoo Finance..!')
st.write(data.describe())

# Preprocess the data
scaler1 = MinMaxScaler(feature_range=(0, 1))
data['Scaled_Open'] = scaler1.fit_transform(np.array(data['Open']).reshape(-1, 1))
data['Scaled_Close'] = scaler1.fit_transform(np.array(data['Close']).reshape(-1, 1))
data['Scaled_High'] = scaler1.fit_transform(np.array(data['High']).reshape(-1, 1))
data['Scaled_Low'] = scaler1.fit_transform(np.array(data['Low']).reshape(-1, 1))

# Define the input and output data
X = np.array(data[['Scaled_Open', 'Scaled_Close', 'Scaled_High', 'Scaled_Low']])
y = np.array(data['Scaled_Close'])

# Define the train and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

#load my LSTM-SVM model
lstm_model = load_model('lstm.h5')
svm_model = joblib.load('svm_model.pkl')

# Make predictions using the LSTM model
lstm_y_train_pred = lstm_model.predict(X_train)
lstm_y_test_pred = lstm_model.predict(X_test)

# Make predictions using the combined LSTM-SVM model
combined_y_train_pred = svm_model.predict(lstm_y_train_pred)
combined_y_test_pred = svm_model.predict(lstm_y_test_pred)

# Rescale the predicted and actual stock prices
lstm_y_train_pred_rescaled = scaler1.inverse_transform(lstm_y_train_pred)
lstm_y_test_pred_rescaled = scaler1.inverse_transform(lstm_y_test_pred)
y_train_rescaled = scaler1.inverse_transform(y_train.reshape(-1, 1))
y_test_rescaled = scaler1.inverse_transform(y_test.reshape(-1, 1))

# Plot the actual and predicted stock prices using the LSTM-SVM model
st.subheader('Prediction using LSTM-SVM Model')
plt.plot(y_test_rescaled, color='red', label='Actual Stock Price')
plt.plot(lstm_y_test_pred_rescaled, color='blue', label='Predicted Stock Price')
plt.title('LSTM-SVM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Calculate the RMSE
rmse = np.sqrt(np.mean((lstm_y_test_pred_rescaled - y_test_rescaled) ** 2))
st.write("RMSE:", rmse)


# Load the CNN-LSTM model
model = load_model('keras_model.h5')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Rescale the predicted and actual stock prices
y_pred_rescaled = scaler1.inverse_transform(y_pred)
y_test_rescaled = scaler1.inverse_transform(y_test.reshape(-1, 1))


# Plot the actual and predicted stock prices using the CNN-LSTM model
st.subheader('Prediction using CNN-LSTM Model')
plt.plot(y_test_rescaled, color='red', label='Actual Stock Price')
plt.plot(y_pred_rescaled, color='blue', label='Predicted Stock Price')
plt.title('CNN-LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Make predictions on the test set
X_pred = np.array([X_test[-1]])
X_pred = X_pred.reshape((X_pred.shape[0], X_pred.shape[1], 1))
y_pred = model.predict(X_pred)

# Inverse transform the scaled prediction
y_pred = scaler1.inverse_transform(y_pred)

# Calculate the RMSE
rmse = np.sqrt(np.mean((y_pred_rescaled - y_test_rescaled) ** 2))
st.write("RMSE:", rmse)

# Streamlit app code
st.write("Tomorrow's predicted closing price:", y_pred[0][0])













