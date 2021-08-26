from sklearn.metrics import mean_absolute_error as mae
from math import sqrt
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt


# ignore warnings
warnings.filterwarnings('ignore')

# style of plot used for the program
style.use('ggplot')

# upload 2014-2018 data to train our model
df = pd.read_csv("TRAINKERO.csv")
print(df.head())

# upload 2019-2021 data to test our model on
test_df = pd.read_csv("TESTKERO.csv")
print(test_df.head())


# sort by date
df = df.sort_values('Date')
test_df = test_df.sort_values('Date')

# fixing the date
df.reset_index(inplace=True)
df.set_index("Date", inplace=True)
test_df.reset_index(inplace=True)
test_df.set_index("Date", inplace=True)
df.tail()

# Visualisation of the training data:

plt.figure(figsize=(12, 6))
plt.plot(df["Price"])
plt.xlabel('Date', fontsize=15)
plt.ylabel('Adjusted Close Price', fontsize=15)
plt.show()


# Setting a Rolling mean for data observation
close_px = df['Price']
mavg = close_px.rolling(window=100).mean()
plt.figure(figsize=(12, 6))
close_px.plot(label='US Gulf Coast Kerosene Jet fuel')
mavg.plot(label='100 day mavg')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()


# change dates into integers for the training of the code
dates_df = df.copy()
dates_df = dates_df.reset_index()

# Store original dates to plot the predicitons
org_dates = dates_df['Date']

# convert dates to integers
dates_df['Date'] = dates_df['Date'].map(mdates.date2num)
dates_df.tail()

# Normalisation of the data

# Creation of the training set of prices:
train_data = df.loc[:, 'Price'].to_numpy()
print(train_data.shape)  # 1258


# Applying normalisation before feeding to LSTM with sklearn:
scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)

scaler.fit(train_data)
train_data = scaler.transform(train_data)

'''Function to create a dataset to feed into an LSTM'''


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# Create the datasets to train our model:
time_steps = 100
X_train, y_train = create_dataset(train_data, time_steps)

# Reshaping the training dataset it [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 100, 1))
print(X_train.shape)


# Visualisation of the datasets :
print('X_train:')
print(str(scaler.inverse_transform(X_train[0])))
print("\n")
print('y_train: ' +
      str(scaler.inverse_transform(y_train[0].reshape(-1, 1)))+'\n')


# Building the LSTM model
model = keras.Sequential()

model.add(LSTM(units=100, return_sequences=True,
          input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the model to our Training set
history = model.fit(X_train, y_train, epochs=20,
                    batch_size=10, validation_split=.30)

# Plotting training & validation loss reults:
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Get the  prices for 2019 to have our model make the predictions
test_data = test_df['Price'].values
test_data = test_data.reshape(-1, 1)
test_data = scaler.transform(test_data)

# Create the data to test our model on:
time_steps = 100
X_test, y_test = create_dataset(test_data, time_steps)

# storing the original values for plotting the predictions
y_test = y_test.reshape(-1, 1)
org_y = scaler.inverse_transform(y_test)

# reshape it [samples, time steps, features]
X_test = np.reshape(X_test, (X_test.shape[0], 100, 1))

# Predict the prices with model:
predicted_y = model.predict(X_test)
predicted_y = scaler.inverse_transform(predicted_y)


# plotting the results:
plt.figure(figsize=(12, 6))
plt.plot(org_y, color='red', label='Real Kerosene Price')
plt.plot(predicted_y, color='blue', label='Predicted Kerosene Price')
plt.title('Kerosene Prices Prediction - LSTM')
plt.xlabel('Time')
plt.ylabel('Price ($/gal)')
plt.legend(loc="lower left")
plt.show()

print(predicted_y)

# Calculation of model performance


# MAPE
def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


mape(org_y, predicted_y)

# RMSE

rms = sqrt(mean_squared_error(org_y, predicted_y))
print(rms)

# MAE
mae(org_y, predicted_y)
