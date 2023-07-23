# ------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
file_test = pd.read_csv('data_test_def.csv', dtype="float64")
file_train = pd.read_csv('data_train_def.csv', dtype="float64")

descargas_test = open('descargas_test.txt','r')
descargas_train = open('descargas_train.txt','r')

max_init = 0
max_fin = 0
matrices = []
nlpca_e_v = []
nlpca_dens = []
print('test')
for n,d in enumerate(descargas_test):
    print(n)
    file_temporal = file_test.iloc[max_init:]
    time = file_temporal['time'].tolist()
    for c,t in enumerate(time):
        if c>0:
            if time[c-1]<=t:
                max_fin+=1
            else:
                break

    mat = file_test.iloc[max_init:max_fin+max_init+1]
    dens = mat['Densidad2_'].tolist()
    mat2 = mat.drop(columns=['time','Densidad2_'])

    nlpca = KernelPCA(n_components=8, kernel='rbf', copy_X=False)  # Configurar el modelo con el número deseado de componentes y el kernel adecuado
    nlpca_features = nlpca.fit_transform(mat2)

    correlaciones = np.corrcoef(nlpca_features.T, dens)
    correlacion_principal = correlaciones[-1, :-1]  # Última fila, todas las columnas excepto la última

    loadings = nlpca.eigenvectors_

    loadings = mat2.values.T.dot(loadings)
    explained_variance_ = nlpca.eigenvalues_ / sum(nlpca.eigenvalues_)

    nlpca_loadings = [f'PC{i}' for i in list(range(1, len(loadings) + 1))]

    loadings_df = pd.DataFrame.from_dict(dict(zip(nlpca_loadings, loadings)))
    loadings_df.index.name = 'feature_names'
    loadings_df.index = ['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
    matrices.append(loadings_df)
    nlpca_e_v.append(np.array(explained_variance_))
    nlpca_dens.append(correlacion_principal)

    max_init += max_fin+1
    max_fin = 0

max_init = 0
max_fin = 0
print('train')
for n,d in enumerate(descargas_train):
    print(n)
    file_temporal = file_train.iloc[max_init:]
    time = file_temporal['time'].tolist()
    for c,t in enumerate(time):
        if c>0:
            if time[c-1]<=t:
                max_fin+=1
            else:
                break

    mat = file_train.iloc[max_init:max_fin+max_init+1]
    dens = mat['Densidad2_'].tolist()
    mat2 = mat.drop(columns=['time','Densidad2_'])

    nlpca = KernelPCA(n_components=8, kernel='rbf',copy_X=False)  # Configurar el modelo con el número deseado de componentes y el kernel adecuado
    nlpca_features = nlpca.fit_transform(mat2)

    correlaciones = np.corrcoef(nlpca_features.T, dens)
    correlacion_principal = correlaciones[-1, :-1]  # Última fila, todas las columnas excepto la última

    loadings = nlpca.eigenvectors_

    loadings = mat2.values.T.dot(loadings)
    explained_variance_ = nlpca.eigenvalues_ / sum(nlpca.eigenvalues_)

    nlpca_loadings = [f'PC{i}' for i in list(range(1, len(loadings) + 1))]

    loadings_df = pd.DataFrame.from_dict(dict(zip(nlpca_loadings, loadings)))
    loadings_df.index.name = 'feature_names'
    loadings_df.index = ['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
    matrices.append(loadings_df)
    nlpca_e_v.append(np.array(explained_variance_))
    nlpca_dens.append(correlacion_principal)

    max_init += max_fin+1
    max_fin = 0

suma_dataframes = None
num_dataframes = 0
for df in matrices:
    if suma_dataframes is None:
        suma_dataframes = df
    else:
        suma_dataframes += df
    num_dataframes += 1

# Calcula el promedio dividiendo el dataframe acumulativo entre el número de dataframes
promedio_dataframe = suma_dataframes / num_dataframes

# Si deseas obtener el promedio como un nuevo dataframe
# promedio_dataframe = pd.DataFrame.mean(suma_dataframes)

print(promedio_dataframe)

arreglo_listas = np.array(nlpca_e_v)
arreglo_listas_dens = np.array(nlpca_dens)
vector_promedio = np.mean(arreglo_listas, axis=0)
vector_promedio_dens = np.mean(arreglo_listas_dens, axis=0)
print(vector_promedio)

print("Correlaciones:")
for i, correlacion in enumerate(vector_promedio_dens):
    print(f"Componente Principal {i+1}: {correlacion}")


plt.bar(range(1,len(vector_promedio)+1),vector_promedio)
plt.plot(range(1,len(vector_promedio)+1),np.cumsum(vector_promedio),c='red',label='Cumulative Explained Variance')

plt.legend(loc='upper left')
plt.xlabel('Number of components')
plt.ylabel('Explained variance (eignenvalues)')
plt.title('Scree plot')

plt.show()
