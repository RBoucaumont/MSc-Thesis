from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggpplot")


# Loading data
df = pd.read_csv("WTI.csv")
print(df.shape)
print(df.head())

#price to predict
actual_price = df.tail(1)
print(actual_price)

# Preparing  data for model
df = df.head(len(df)-1)
print(df.shape)

# Creating empty list to store independent and dependent variables
days = list()
adj_close_prices = list()

df_days = df.loc[:, "Date"]
df_adj_close = df.loc[:, "Price"]


# creating idenpendant dataset
for day in df_days:
    days.append([int(day.split('/')[0])])

# creating  dependent dataset
for adj_close_price in df_adj_close:
    adj_close_prices.append(float(adj_close_price))

# Data prices observations
print(days)
print(adj_close_prices)

len(days)
len(adj_close_prices)


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
plt.figure(figsize=(12, 6))
plt.scatter(days, adj_close_prices, color="red", label="Data")
plt.plot(days, rbf_svr.predict(days), color="green", label="RBF Model")
plt.plot(days, poly_svr.predict(days),
         color="orange", label="Polynomial Model")
plt.plot(days, lin_svr.predict(days), color="blue", label="Linear Model")
plt.title("SVR model comparison")
plt.ylabel("Prices")
plt.xlabel("Dates")
plt.legend()
plt.plot()


# Show predicted price for given day
day = [[31]]
print("The RBF SVR Predicted:", rbf_svr.predict(day))
print("The Linear SVR Predicted:", lin_svr.predict(day))
print("The Poly SVR Predicted:", poly_svr.predict(day))


# Print actual price of stock on day 31
print("Acual price:", actual_price['Price'][21])
