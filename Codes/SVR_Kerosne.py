from sklearn.metrics import mean_absolute_error as mae
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np
from matplotlib import style

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Style the plot 
style.use('ggplot')


# get 2019-2021 data to test our model on
test_df = pd.read_csv("TESTKERO.csv")
print(test_df.head())

# sort by date
#df = df.sort_values('Date')
test_df = test_df.sort_values('Date')

# fixing dates
test_df.reset_index(inplace=True)
test_df.set_index("Date", inplace=True)

# visualisation of datasets
plt.figure(figsize=(12, 6))
plt.plot(test_df["Price"])
plt.xlabel('Date', fontsize=15)
plt.ylabel('Adjusted Close Price', fontsize=15)
plt.show()


# Rolling mean for observation
close_px = test_df['Price']
mavg = close_px.rolling(window=100).mean()

plt.figure(figsize=(12, 6))
close_px.plot(label='kero')
mavg.plot(label='mavg')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()


# change dates into integers for training
dates_df = test_df.copy()
dates_df = dates_df.reset_index()

# Store  original dates for plotting predicitons
org_dates = dates_df['Date']

# convert to integers
dates_df['Date'] = dates_df['Date'].map(mdates.date2num)
dates_df.tail()

# Use sklearn support vector regression to predict our data:

dates = dates_df['Date'].to_numpy()
prices = test_df['Price'].to_numpy()

# Convert to 1 dimensional  Vector
dates = np.reshape(dates, (len(dates), 1))
prices = np.reshape(prices, (len(prices), 1))
# rbf model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(dates, prices)

#plotting the visualisation of the prediction 

plt.figure(figsize=(12, 6))
plt.plot(dates, prices, color='red', label='Real Kerosene Price')
plt.plot(dates, svr_rbf.predict(dates), color='blue',
         label='Predicted RBF SVR Price')
plt.ylabel('Price ($/gal)')
plt.title("Kerosene Prices Prediction - SVR")
plt.legend()
plt.show()


# calculation of forecasting performance

# MAPE
def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


svr_predicted = svr_rbf.predict(dates)
# print(svr_predicted)
mape(prices, svr_predicted)

# RMSE
rms = sqrt(mean_squared_error(prices, svr_predicted))
print(rms)


#  MAE
mae(prices, svr_predicted)
