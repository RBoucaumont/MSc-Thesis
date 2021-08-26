from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Load the data
df = pd.read_csv("1MO_KERO.csv")
print(df.shape)
print(df.head())

# actual price to predict

actual_price = df.tail(1)
print(actual_price)

# Prepare the data for models

df = df.head(len(df)-1)
print(df.shape)

# Create empty list to store independent and dependent data
days = list()
adj_close_prices = list()

df_days = df.loc[:, "Date"]
df_adj_close = df.loc[:, "Price"]

# create the idenpendant dataset
for day in df_days:
    days.append([int(day.split('/')[0])])

    # create the dependent dataset
for adj_close_price in df_adj_close:
    adj_close_prices.append(float(adj_close_price))

# Print days and adj close price
print(days)
print(adj_close_prices)

# create the 3 support Vector Regression models
# linear model
lin_svr = SVR(kernel="linear", C=1000.0)
lin_svr.fit(days, adj_close_prices)

# polynomial model
poly_svr = SVR(kernel="poly", C=1000.0, degree=2, gamma=0.1)
poly_svr.fit(days, adj_close_prices)

# RBF model
rbf_svr = SVR(kernel="rbf", C=1000.0, gamma=0.1)
rbf_svr.fit(days, adj_close_prices)

# plot models to see which has the best fit to the original data
plt.figure(figsize=(16, 8))
plt.scatter(days, adj_close_prices, color="red", label="Data")
plt.plot(days, rbf_svr.predict(days), color="green", label="RBF Model")
plt.plot(days, poly_svr.predict(days),
         color="orange", label="Polynomial Model")
plt.plot(days, lin_svr.predict(days), color="blue", label="Linear Model")
plt.legend()
plt.plot()
