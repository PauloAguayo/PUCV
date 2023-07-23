% Datos de entrada (series temporales multivariadas)
data = importdata('datos.csv');  % Asegúrate de tener tus datos en un archivo CSV o ajusta la carga de datos según corresponda

% Crear el modelo ARMAX
p = 1;  % Orden autorregresivo
q = 9;  % Orden de media móvil
k = 9;  % Cantidad de variables exógenas
model = armax(data, [p q k, ]jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj);

% Ajustar el modelo a los datos
results = estimate(model, data);

% Obtener los parámetros estimados del modelo
params = results.AR;

% Realizar predicciones con el modelo ajustado
predictions = forecast(model, data, 10);  % Predicción de 10 pasos hacia adelante

% Visualizar los resultados
plot(data);
hold on;
plot(predictions);
legend('Datos originales', 'Predicciones');
