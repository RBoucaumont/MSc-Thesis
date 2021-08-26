from sklearn.metrics import mean_absolute_error as mae
from math import sqrt
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import DataFrame
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib import style
style.use('ggplot')
warnings.filterwarnings("ignore")

# observe dataset
df = pd.read_csv("ARMA Kerosene.csv")
df.head(5)

# find correlation in dataset
plt.figure()
lag_plot(df['Price'], lag=3)
plt.title('Kerosene - Autocorrelation plot with lag = 3')
plt.show()

# plot Kero prices
plt.plot(df["Date"], df["Price"])
plt.xticks(np.arange(0, 2140, 500), df['Date'][0:2140:500])
plt.title("Kerosene  price over time")
plt.xlabel("time")
plt.ylabel("price")
plt.show()

# split and train the dataset
train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
training_data = train_data['Price'].values
test_data = test_data['Price'].values
history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)
for time_point in range(N_test_observations):
    model = ARIMA(history, order=(4, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)
MSE_error = mean_squared_error(test_data, model_predictions)
print('Testing Mean Squared Error is {}'.format(MSE_error))

# observing and finding PACF
plot_pacf(history, lags=50)


# summary of fit model
print(model_fit.summary())


# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.title("Residuals - Stationary")
plt.show()


# density plot of residuals
residuals.plot(kind='kde')
plt.title("Density plot of residuals")

# plot forecasting results
plt.figure(figsize=(12, 6))
test_set_range = df[int(len(df)*0.7):].index
plt.plot(test_set_range, test_data, color='red', label='Real Kerosene Price')
plt.plot(test_set_range, model_predictions,
         color='blue', label='Predicted ARMA Price')

plt.title('Kerosene Prices Prediction - ARMA')
plt.ylabel('Prices ($/gal)')
plt.xticks(np.arange(1700, 2140, 100), df['Date'][1700:2140:100])
plt.legend()
plt.show()


# summary stats of residuals
print(residuals.describe())

# Models performance calculations

#MAPE
def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


mape(test_data, model_predictions)

#RMS
rms = sqrt(mean_squared_error(test_data, model_predictions))
print(rms)


# MAE
mae(test_data, model_predictions)
