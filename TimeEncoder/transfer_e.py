# ------------------------------------------------------------------------------
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
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
matrices_ssrftest = {'ACTON275':{}, 'BOL5':{}, 'ECE7':{},'GR':{},'GR2':{},'HALFAC3':{},'IACCEL1':{},'RX306':{}}
temp_w = 100
target = 'Densidad2_'

n_mat_1 = 0
n_mat_2 = 0
print('test')
for d in descargas_test:
    print(n_mat_1)
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

    for var in ['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']:
        data = mat2[[target, var]]
        granger_results = grangercausalitytests(data, maxlag=temp_w, verbose=False)

        for lag in range(1, temp_w+1):

            if n_mat_1==0:
                matrices_ssrftest[var][str(lag)] = [list(granger_results[lag][0]['ssr_ftest'][:2])]
            else:
                matrices_ssrftest[var][str(lag)].append(list(granger_results[lag][0]['ssr_ftest'][:2]))

    max_init = max_fin+max_init+1
    max_fin = 0
    n_mat_1 +=1

max_init = 0
max_fin = 0
print('train')
for d in descargas_train:
    print(n_mat_2)
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

    for var in ['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']:
        data = mat2[[target, var]]
        granger_results = grangercausalitytests(data, maxlag=temp_w, verbose=False)

        for lag in range(1, temp_w+1):

            matrices_ssrftest[var][str(lag)].append(list(granger_results[lag][0]['ssr_ftest'][:2]))

    max_init = max_fin+max_init+1
    max_fin = 0
    n_mat_2 +=1
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def_ = {'ACTON275':{}, 'BOL5':{}, 'ECE7':{},'GR':{},'GR2':{},'HALFAC3':{},'IACCEL1':{},'RX306':{}}
for var in ['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']:
    for lag in range(1, temp_w+1):
        matrices_ssrftest[var][str(lag)] = np.array(matrices_ssrftest[var][str(lag)])
        matrices_ssrftest[var][str(lag)] = pd.DataFrame(matrices_ssrftest[var][str(lag)].reshape(-1,2))
        matrices_ssrftest[var][str(lag)] = matrices_ssrftest[var][str(lag)].mean()

        def_[var][str(lag)] = {}

        def_[var][str(lag)]['t_e'] = matrices_ssrftest[var][str(lag)][0] # CORREGIR !!!!!!
        def_[var][str(lag)]['p-val'] = matrices_ssrftest[var][str(lag)][1]



# df = pd.DataFrame.from_dict(matrices_ssrftest, orient='columns')
df = pd.DataFrame.from_dict(def_, orient='columns')

# Export the DataFrame to an Excel file
df.to_excel('transfer_entropy.xlsx', index=True)
# Guardar cada DataFrame en una hoja de cálculo separada
# df_matrices_ssrftest = pd.concat(matrices_ssrftest.values(), axis=1, keys=matrices_ssrftest.keys())
#
# df_matrices_ssrftest.to_excel(writer, sheet_name='ssrftest', index=False)
#
# # Cerrar el objeto ExcelWriter
# writer.save()
