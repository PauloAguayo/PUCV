import pandas as pd
import statsmodels.api as sm
from utils import data_loading_decom


n_signal = 3
seq_len = 9

# Cargar los datos de la serie temporal multivariada en un DataFrame
data = pd.read_csv('data_test_def.csv')
data_train = data_loading_decom(data.values, seq_len=seq_len, n_signal=[n_signal])
d = list(data.columns)
d.remove('Densidad2_')


# Realizar el análisis de descomposición
for i in data_train[1]:
    # Generar el rango de fechas con la frecuencia en milisegundos
    dates = pd.date_range(start='2023-01-01', periods=len(i[0]), freq='D')
    da = pd.Series(i[0],index=dates)#, columns=d)
    # da['time'] = dates

    # da['time'] = pd.to_datetime(da['time'], unit='D')

    # Establecer la columna 'Fecha' como el índice del DataFrame
    # da.set_index('time', inplace=True)
    print(da)


    decomposition = sm.tsa.seasonal_decompose(da, model='additive')

    # Obtener los componentes descompuestos
    trend = decomposition.trend
    seasonality = decomposition.seasonal
    residuals = decomposition.resid

    # Visualizar los componentes descompuestos
    trend.plot()
    seasonality.plot()
    residuals.plot()

    # Calcular y visualizar la serie temporal reconstruida
    reconstructed = trend + seasonality + residuals
    reconstructed.plot()

    # Calcular y visualizar la serie de residuos
    residuals.plot()
