# ------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
file_test = pd.read_csv('data_test_def.csv')
file_train = pd.read_csv('data_train_def.csv')

descargas_test = open('descargas_test.txt','r')
descargas_train = open('descargas_train.txt','r')

max_init = 0
max_fin = 0
matrices = []
pca_e_v = []
n_mat = 0
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
    mat2 = mat.drop(columns=['time','Densidad2_'])

    pca = PCA(n_components=3)
    pca.fit_transform(mat2)

    pca_features=pca.fit_transform(mat2)


    loadings = pca.components_
    n_features = pca.n_features_


    pca_loadings = [f'PC{i}' for i in list(range(1, len(loadings) + 1))]
    pca_df = pd.DataFrame(data=pca_features, columns=pca_loadings)

    loadings_df = pd.DataFrame.from_dict(dict(zip(pca_loadings, loadings)))
    loadings_df.index.name = 'feature_names'
    loadings_df.index = ['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
    matrices.append(loadings_df)
    pca_e_v.append(np.array(pca.explained_variance_.tolist()))
    # break

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
    mat2 = mat.drop(columns=['time','Densidad2_'])

    pca = PCA(n_components=3)
    pca.fit_transform(mat2)

    pca_features=pca.fit_transform(mat2)


    loadings = pca.components_
    n_features = pca.n_features_


    pca_loadings = [f'PC{i}' for i in list(range(1, len(loadings) + 1))]
    pca_df = pd.DataFrame(data=pca_features, columns=pca_loadings)

    loadings_df = pd.DataFrame.from_dict(dict(zip(pca_loadings, loadings)))
    loadings_df.index.name = 'feature_names'
    loadings_df.index = ['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
    matrices.append(loadings_df)
    pca_e_v.append(np.array(pca.explained_variance_.tolist()))

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

arreglo_listas = np.array(pca_e_v)
vector_promedio = np.mean(arreglo_listas, axis=0)
print(vector_promedio)
plt.bar(range(1,len(vector_promedio)+1),vector_promedio)
plt.plot(range(1,len(vector_promedio)+1),np.cumsum(vector_promedio),c='red',label='Cumulative Explained Variance')

plt.legend(loc='upper left')
plt.xlabel('Number of components')
plt.ylabel('Explained variance (eignenvalues)')
plt.title('Scree plot')

plt.show()
