# ------------------------------------------------------------------------------
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
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
matrices_ssrftest = {'ACTON275':{}, 'BOL5':{}, 'ECE7':{},'GR':{},'GR2':{},'HALFAC3':{},'IACCEL1':{},'RX306':{}}
matrices_ftest = {'ACTON275':{}, 'BOL5':{}, 'ECE7':{},'GR':{},'GR2':{},'HALFAC3':{},'IACCEL1':{},'RX306':{}}
matrices_ssrchi = {'ACTON275':{}, 'BOL5':{}, 'ECE7':{},'GR':{},'GR2':{},'HALFAC3':{},'IACCEL1':{},'RX306':{}}
matrices_lrtest = {'ACTON275':{}, 'BOL5':{}, 'ECE7':{},'GR':{},'GR2':{},'HALFAC3':{},'IACCEL1':{},'RX306':{}}
temp_w = 20
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
                matrices_ftest[var][str(lag)] = [list(granger_results[lag][0]['params_ftest'][:2])]
                matrices_ssrftest[var][str(lag)] = [list(granger_results[lag][0]['ssr_ftest'][:2])]
                matrices_ssrchi[var][str(lag)] = [list(granger_results[lag][0]['ssr_chi2test'][:2])]
                matrices_lrtest[var][str(lag)] = [list(granger_results[lag][0]['lrtest'][:2])]
            else:
                matrices_ftest[var][str(lag)].append(list(granger_results[lag][0]['params_ftest'][:2]))
                matrices_ssrftest[var][str(lag)].append(list(granger_results[lag][0]['ssr_ftest'][:2]))
                matrices_ssrchi[var][str(lag)].append(list(granger_results[lag][0]['ssr_chi2test'][:2]))
                matrices_lrtest[var][str(lag)].append(list(granger_results[lag][0]['lrtest'][:2]))

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
            matrices_ftest[var][str(lag)].append(list(granger_results[lag][0]['params_ftest'][:2]))
            matrices_ssrftest[var][str(lag)].append(list(granger_results[lag][0]['ssr_ftest'][:2]))
            matrices_ssrchi[var][str(lag)].append(list(granger_results[lag][0]['ssr_chi2test'][:2]))
            matrices_lrtest[var][str(lag)].append(list(granger_results[lag][0]['lrtest'][:2]))

    max_init = max_fin+max_init+1
    max_fin = 0
    n_mat_2 +=1
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def1 = {'ACTON275':{}, 'BOL5':{}, 'ECE7':{},'GR':{},'GR2':{},'HALFAC3':{},'IACCEL1':{},'RX306':{}}
def2 = {'ACTON275':{}, 'BOL5':{}, 'ECE7':{},'GR':{},'GR2':{},'HALFAC3':{},'IACCEL1':{},'RX306':{}}
def3 = {'ACTON275':{}, 'BOL5':{}, 'ECE7':{},'GR':{},'GR2':{},'HALFAC3':{},'IACCEL1':{},'RX306':{}}
def4 = {'ACTON275':{}, 'BOL5':{}, 'ECE7':{},'GR':{},'GR2':{},'HALFAC3':{},'IACCEL1':{},'RX306':{}}
for var in ['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']:
    for lag in range(1, temp_w+1):
        matrices_ftest[var][str(lag)] = np.array(matrices_ftest[var][str(lag)])
        matrices_ssrftest[var][str(lag)] = np.array(matrices_ssrftest[var][str(lag)])
        matrices_ssrchi[var][str(lag)] = np.array(matrices_ssrchi[var][str(lag)])
        matrices_lrtest[var][str(lag)] = np.array(matrices_lrtest[var][str(lag)])

        matrices_ftest[var][str(lag)] = pd.DataFrame(matrices_ftest[var][str(lag)].reshape(-1,2))
        matrices_ssrftest[var][str(lag)] = pd.DataFrame(matrices_ssrftest[var][str(lag)].reshape(-1,2))
        matrices_ssrchi[var][str(lag)] = pd.DataFrame(matrices_ssrchi[var][str(lag)].reshape(-1,2))
        matrices_lrtest[var][str(lag)] = pd.DataFrame(matrices_lrtest[var][str(lag)].reshape(-1,2))

        matrices_ftest[var][str(lag)] = matrices_ftest[var][str(lag)].mean()
        matrices_ssrftest[var][str(lag)] = matrices_ssrftest[var][str(lag)].mean()
        matrices_ssrchi[var][str(lag)] = matrices_ssrchi[var][str(lag)].mean()
        matrices_lrtest[var][str(lag)] = matrices_lrtest[var][str(lag)].mean()

        def1[var][str(lag)] = {}
        def1[var][str(lag)]['test'] = matrices_ftest[var][str(lag)][0]
        def1[var][str(lag)]['p-val'] = matrices_ftest[var][str(lag)][1]

        def2[var][str(lag)] = {}
        def2[var][str(lag)]['test'] = matrices_ssrftest[var][str(lag)][0]
        def2[var][str(lag)]['p-val'] = matrices_ssrftest[var][str(lag)][1]

        def3[var][str(lag)] = {}
        def3[var][str(lag)]['test'] = matrices_ssrchi[var][str(lag)][0]
        def3[var][str(lag)]['p-val'] = matrices_ssrchi[var][str(lag)][1]

        def4[var][str(lag)] = {}
        def4[var][str(lag)]['test'] = matrices_lrtest[var][str(lag)][0]
        def4[var][str(lag)]['p-val'] = matrices_lrtest[var][str(lag)][1]


df1 = pd.DataFrame.from_dict(def1, orient='columns')
df2 = pd.DataFrame.from_dict(def2, orient='columns')
df3 = pd.DataFrame.from_dict(def3, orient='columns')
df4 = pd.DataFrame.from_dict(def4, orient='columns')

writer = pd.ExcelWriter('granger.xlsx', engine='xlsxwriter')

df1.to_excel(writer, sheet_name='ftest', index=True)
df2.to_excel(writer, sheet_name='ssrftest', index=True)
df1.to_excel(writer, sheet_name='ssrchi', index=True)
df2.to_excel(writer, sheet_name='lrtest', index=True)

# Save the Excel file
writer.save()
