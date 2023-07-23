# ------------------------------------------------------------------------------
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
file_test = pd.read_csv('data_test_def.csv')
file_train = pd.read_csv('data_train_def.csv')

descargas_test = open('descargas_test.txt','r')
descargas_train = open('descargas_train.txt','r')

max_init = 0
max_fin = 0
matrices_corr = []
matrices_pval = []
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
    mat2 = mat.drop(columns='time')
    corr, pval = spearmanr(mat2)
    matrices_corr.append(np.array(corr.tolist()))
    matrices_pval.append(np.array(pval.tolist()))
    max_init = max_fin+max_init+1
    max_fin = 0
    n_mat +=1

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
    mat2 = mat.drop(columns='time')
    corr, pval = spearmanr(mat2)
    matrices_corr.append(np.array(corr.tolist()))
    matrices_pval.append(np.array(pval.tolist()))
    max_init = max_fin+max_init+1
    max_fin = 0
    n_mat +=1

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
matrices_corr = np.array(matrices_corr)
matrices_pval = np.array(matrices_pval)
df_corr = pd.DataFrame(matrices_corr.reshape(n_mat,-1))
df_pval = pd.DataFrame(matrices_pval.reshape(n_mat,-1))
promedio_corr = df_corr.mean()
promedio_pval = df_pval.mean()

matriz_promedio_corr = np.array(promedio_corr).reshape(corr.shape)
matriz_promedio_pval = np.array(promedio_pval).reshape(corr.shape)

df_promedio_corr = pd.DataFrame(matriz_promedio_corr)
df_promedio_pval = pd.DataFrame(matriz_promedio_pval)
df_promedio_corr.columns =['ACTON275', 'BOL5', 'Densidad2_', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
df_promedio_corr.index =['ACTON275', 'BOL5', 'Densidad2_', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
df_promedio_pval.columns =['ACTON275', 'BOL5', 'Densidad2_', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
df_promedio_pval.index =['ACTON275', 'BOL5', 'Densidad2_', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
print('Promedio de correlaciones:')
print(df_promedio_corr)
print('-------------------------------------------------------------------------')
print('Promedio de valores p:')
print(df_promedio_pval)
df_promedio_corr.to_csv('spearman_corr.csv', sep=';', float_format="%.8f")
df_promedio_pval.to_csv('spearman_pval.csv', sep=';', float_format="%.8f")
# ------------------------------------------------------------------------------
