import numpy as np
# import statsmodels.api as sm
from utils import data_loading_decom
import pandas as pd
import statsmodels.tsa.arima.model as sm

data = pd.read_csv('data_train_def.csv')

X, y = data_loading_decom(data.values,[3])

# Definir la estructura del modelo ARMAX
p = 0
d = 1  # Orden AR
q = 9  # Orden MA

# Crear y ajustar el modelo ARMAX para cada serie temporal
n_series = len(X)
models = []


for i in range(n_series):
    X_ = pd.DataFrame(X[i])
    y_ = pd.DataFrame(y[i][0])
    model = sm.ARIMA(y_, exog=X_, order=(p,d,q))
    results = model.fit()
    models.append(results)

print(models)
# Evaluar el rendimiento del modelo
# ...
