# ------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
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
pca_e_v = []
pca_dens = []
pca_p = []
for d in descargas_test:
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

    pca = PCA(n_components=8)
    pca.fit_transform(mat2)

    pca_features=pca.fit_transform(mat2)

    #correlaciones = np.corrcoef(pca_features.T, dens)
    #correlacion_principal = correlaciones[-1, :-1]  # Última fila, todas las columnas excepto la última

    correlacion, p_ = spearmanr(pca_features, dens)

    correlacion_principal = correlacion[-1,:-1]
    p_principal = p_[-1,:-1]

    loadings = pca.components_
    n_features = pca.n_features_

    pca_loadings = [f'PC{i}' for i in list(range(1, len(loadings) + 1))]

    loadings_df = pd.DataFrame.from_dict(dict(zip(pca_loadings, loadings)))
    loadings_df.index.name = 'feature_names'
    loadings_df.index = ['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
    matrices.append(loadings_df)
    pca_e_v.append(np.array(pca.explained_variance_.tolist()))
    pca_dens.append(correlacion_principal)
    pca_p.append(p_principal)

    max_init += max_fin+1
    max_fin = 0

max_init = 0
max_fin = 0
for d in descargas_train:
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

    pca = PCA(n_components=8)
    pca.fit_transform(mat2)

    pca_features=pca.fit_transform(mat2)

    # correlaciones = np.corrcoef(pca_features.T, dens)
    # correlacion_principal = correlaciones[-1, :-1]  # Última fila, todas las columnas excepto la última

    correlacion, p_ = spearmanr(pca_features, dens)

    correlacion_principal = correlacion[-1,:-1]
    p_principal = p_[-1,:-1]

    loadings = pca.components_
    n_features = pca.n_features_

    pca_loadings = [f'PC{i}' for i in list(range(1, len(loadings) + 1))]

    loadings_df = pd.DataFrame.from_dict(dict(zip(pca_loadings, loadings)))
    loadings_df.index.name = 'feature_names'
    loadings_df.index = ['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
    matrices.append(loadings_df)
    pca_e_v.append(np.array(pca.explained_variance_.tolist()))
    pca_dens.append(correlacion_principal)
    pca_p.append(p_principal)

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

print(promedio_dataframe)

arreglo_listas = np.array(pca_e_v)
arreglo_listas_dens = np.array(pca_dens)
arreglo_listas_p = np.array(pca_p)
vector_promedio = np.mean(arreglo_listas, axis=0)
vector_promedio_dens = np.mean(arreglo_listas_dens, axis=0)
vector_promedio_p = np.mean(arreglo_listas_p, axis=0)
print(vector_promedio)

print("Correlaciones:")
for i, (correlacion,v_p) in enumerate(zip(vector_promedio_dens,vector_promedio_p)):
    print(f"Componente Principal {i+1} - Correlacion: {correlacion}, Valor p:{v_p}")


plt.bar(range(1,len(vector_promedio)+1),vector_promedio)
plt.plot(range(1,len(vector_promedio)+1),np.cumsum(vector_promedio),c='red',label='Cumulative Explained Variance')

plt.legend(loc='upper left')
plt.xlabel('Number of components')
plt.ylabel('Explained variance (eignenvalues)')
plt.title('Scree plot')

plt.show()
