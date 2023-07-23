import numpy as np
from utils import data_loading
import pandas as pd
import statsmodels.tsa.arima.model as sm
from CustomLoss import WMAPE
from sklearn.metrics import mean_squared_error

data_train = pd.read_csv('data_train_def.csv')
data_test = pd.read_csv('data_test_def.csv')

X_train, y_train = data_loading(data_train.values,9,[3])
X_test, y_test = data_loading(data_test.values,9,[3])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = X_train.reshape(-1, 72)
X_test = X_test.reshape(-1, 72)


# Definir la estructura del modelo ARMAX
p = 0  # Orden AR
d = 0  # Orden diferencial
q = 0  # Orden MA

# TRAIN
model = sm.ARIMA(y_train, exog=X_train, order=(p,d,q))
results = model.fit()

# TEST
forecast = results.predict(start=1, end=len(y_test)+1,exog=X_test)
forecast = np.array(forecast)
forecast = forecast.reshape(-1,1)
print(forecast, forecast.shape)

loss_1 = WMAPE(y_test, forecast[:-1,:])
loss_2 = mean_squared_error(y_test, forecast[:-1,:])
print("WMAPE Error:", loss_1)
print("Mean Squared Error:", loss_2)

loss_1 = WMAPE(y_test, forecast[1:,:])
loss_2 = mean_squared_error(y_test, forecast[1:,:])
print("WMAPE Error:", loss_1)
print("Mean Squared Error:", loss_2)
