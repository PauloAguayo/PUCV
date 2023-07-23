import pandas as pd
import matplotlib.pyplot as plt

# Datos de ejemplo
data_train = pd.read_csv('data_train_def.csv')
data_test = pd.read_csv('data_test_def.csv')

data_train = data_train['Densidad2_'].tolist()
data_test = data_test['Densidad2_'].tolist()

total = data_train + data_test

# Crear el histograma
plt.hist(data_train, bins=40, alpha=1, label='Train')
plt.hist(data_test, bins=40, alpha=0.6, label='Test')
plt.hist(total, bins=40, alpha=0.3, label='Todos')

# Agregar etiquetas y título
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma')

# Mostrar el histograma
plt.show()
