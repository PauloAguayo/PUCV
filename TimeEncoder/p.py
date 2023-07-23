import numpy as np

# Obtén los autovalores de los componentes principales obtenidos en NLPCA
autovalores = np.array([0.2, 0.15, 0.1, 0.08, 0.05])  # Ejemplo de autovalores

# Ordena los autovalores de forma descendente
autovalores_ordenados = np.sort(autovalores)[::-1]

# Calcula la suma total de todos los autovalores
suma_total = np.sum(autovalores_ordenados)

# Calcula la energía acumulada para cada k (desde 1 hasta el número total de componentes principales)
energia_acumulada = np.cumsum(autovalores_ordenados) / suma_total

# Imprime los resultados
print("Varianza Explicada:")
for k, energia in enumerate(energia_acumulada, 1):
    print(f"Componentes principales {k}: {energia}")
