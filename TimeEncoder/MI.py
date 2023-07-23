# ------------------------------------------------------------------------------
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
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
matrices = []
temp_w = 20
target = 'Densidad2_'
windows = {'ACTON275':{}, 'BOL5':{}, 'ECE7':{},'GR':{},'GR2':{},'HALFAC3':{},'IACCEL1':{},'RX306':{}}


print('test')
nn = 0
for d in descargas_test:
    nn+=1
    print(nn)
    file_temporal = file_test.iloc[max_init:]
    time = file_temporal['time'].tolist()
    for c,t in enumerate(time):
        if c>0:
            if time[c-1]<=t:
                max_fin+=1
            else:
                break

    mat = file_test.iloc[max_init:max_fin+max_init+1]
    mat2 = mat.drop(columns=['time'])
    mat2 = mat2.reset_index()
    for w in range(2,21):
        print(w)
        #dens = mat['Densidad2_'].rolling(window=w)
        #mat2 = mat.drop(columns=['time','Densidad2_'])

        #mat2 = mat2[['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']]
        for i in range(len(mat2) - w + 1):
            m = mat2.iloc[i:i + w]
            mi_scores = mutual_info_regression(m[['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']], m['Densidad2_'], n_neighbors=1)
            for mi,i in zip(mi_scores,['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']):
                if str(w) not in list(windows[i].keys()):
                    windows[i][str(w)] = [mi]
                else:
                    windows[i][str(w)].append(mi)

    max_init = max_fin+max_init+1
    max_fin = 0


max_init = 0
max_fin = 0
print('train')
for d in descargas_train:
    nn+=1
    print(nn)
    file_temporal = file_train.iloc[max_init:]
    time = file_temporal['time'].tolist()
    for c,t in enumerate(time):
        if c>0:
            if time[c-1]<=t:
                max_fin+=1
            else:
                break

    mat = file_train.iloc[max_init:max_fin+max_init+1]
    mat2 = mat.drop(columns=['time'])
    mat2 = mat2.reset_index()
    for w in range(2,21):
        print(w)
        #dens = mat['Densidad2_'].rolling(window=w)
        #mat2 = mat.drop(columns=['time','Densidad2_'])

        #mat2 = mat2[['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']]
        for i in range(len(mat2) - w + 1):
            m = mat2.iloc[i:i + w]
            mi_scores = mutual_info_regression(m[['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']], m['Densidad2_'], n_neighbors=1)
            for mi,i in zip(mi_scores,['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']):
                if str(w) not in list(windows[i].keys()):
                    windows[i][str(w)] = [mi]
                else:
                    windows[i][str(w)].append(mi)

    max_init = max_fin+max_init+1
    max_fin = 0

print(windows)
for k in ['ACTON275', 'BOL5', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']:
    for j in windows[k].keys():
        windows[k][j] = sum(windows[k][j])/len(windows[k][j])
print(windows)

df = pd.DataFrame.from_dict(windows, orient='columns')

writer = pd.ExcelWriter('MI.xlsx', engine='xlsxwriter')

df.to_excel(writer,  index=True)

writer.save()
    #
    # max_init = 0
    # max_fin = 0
    # for d in descargas_train:
    #     file_temporal = file_train.iloc[max_init:]
    #     time = file_temporal['time'].tolist()
    #     for c,t in enumerate(time):
    #         if c>0:
    #             if time[c-1]<=t:
    #                 max_fin+=1
    #             else:
    #                 break
    #
    #     mat = file_train.iloc[max_init:max_fin+max_init+1]
    #     mat2 = mat.drop(columns='time')
    #
    #     data = mat2[[target, var]]
    #     granger_results = grangercausalitytests(data, maxlag=temp_w, verbose=False)
    #
    #     for lag in range(1, temp_w+1):
    #         matrices_ftest[str(lag)].append(list(granger_results[lag][0]['params_ftest'][:2]))
    #         matrices_ssrftest[str(lag)].append(list(granger_results[lag][0]['ssr_ftest'][:2]))
    #         matrices_ssrchi[str(lag)].append(list(granger_results[lag][0]['ssr_chi2test'][:2]))
    #         matrices_lrtest[str(lag)].append(list(granger_results[lag][0]['lrtest'][:2]))
    #     max_init = max_fin+max_init+1
    #     max_fin = 0
    #     n_mat_2 +=1
    # # ------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------
    # for lag in range(1, temp_w+1):
    #     matrices_ftest[str(lag)] = np.array(matrices_ftest[str(lag)])
    #     matrices_ssrftest[str(lag)] = np.array(matrices_ssrftest[str(lag)])
    #     matrices_ssrchi[str(lag)] = np.array(matrices_ssrchi[str(lag)])
    #     matrices_lrtest[str(lag)] = np.array(matrices_lrtest[str(lag)])
    #
    #     matrices_ftest[str(lag)] = pd.DataFrame(matrices_ftest[str(lag)].reshape(-1,2))
    #     matrices_ssrftest[str(lag)] = pd.DataFrame(matrices_ssrftest[str(lag)].reshape(-1,2))
    #     matrices_ssrchi[str(lag)] = pd.DataFrame(matrices_ssrchi[str(lag)].reshape(-1,2))
    #     matrices_lrtest[str(lag)] = pd.DataFrame(matrices_lrtest[str(lag)].reshape(-1,2))
    #
    #     matrices_ftest[str(lag)] = matrices_ftest[str(lag)].mean()
    #     matrices_ssrftest[str(lag)] = matrices_ssrftest[str(lag)].mean()
    #     matrices_ssrchi[str(lag)] = matrices_ssrchi[str(lag)].mean()
    #     matrices_lrtest[str(lag)] = matrices_lrtest[str(lag)].mean()
    #
    #
    #
    #
    # # Guardar cada DataFrame en una hoja de cálculo separada
    # df_matrices_ftest = pd.concat(matrices_ftest.values(), axis=1, keys=matrices_ftest.keys())
    # df_matrices_ssrftest = pd.concat(matrices_ssrftest.values(), axis=1, keys=matrices_ssrftest.keys())
    # df_matrices_ssrchi = pd.concat(matrices_ssrchi.values(), axis=1, keys=matrices_ssrchi.keys())
    # df_matrices_lrtest = pd.concat(matrices_lrtest.values(), axis=1, keys=matrices_lrtest.keys())
    #
    # df_matrices_ftest.to_excel(writer, sheet_name='ftest', index=False)
    # df_matrices_ssrftest.to_excel(writer, sheet_name='ssrftest', index=False)
    # df_matrices_ssrchi.to_excel(writer, sheet_name='ssrchi', index=False)
    # df_matrices_lrtest.to_excel(writer, sheet_name='lrtest', index=False)
    #
    # # Cerrar el objeto ExcelWriter
    # writer.save()
    #     # matriz_promedio_corr = np.array(promedio_corr).reshape(corr.shape)
    #     # matriz_promedio_pval = np.array(promedio_pval).reshape(corr.shape)
    #
    #     # df_promedio_corr = pd.DataFrame(matriz_promedio_corr)
    #     # df_promedio_pval = pd.DataFrame(matriz_promedio_pval)
    #     # df_promedio_corr.columns =['ACTON275', 'BOL5', 'Densidad2_', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
    #     # df_promedio_corr.index =['ACTON275', 'BOL5', 'Densidad2_', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
    #     # df_promedio_pval.columns =['ACTON275', 'BOL5', 'Densidad2_', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
    #     # df_promedio_pval.index =['ACTON275', 'BOL5', 'Densidad2_', 'ECE7','GR','GR2','HALFAC3','IACCEL1','RX306']
    #     # print('Promedio de correlaciones:')
    #     # print(df_promedio_corr)
    #     # print('-------------------------------------------------------------------------')
    #     # print('Promedio de valores p:')
    #     # print(df_promedio_pval)
    #     # df_promedio_corr.to_csv('spearman_corr.csv')
    #     # df_promedio_pval.to_csv('spearman_pval.csv')
    #     # # ------------------------------------------------------------------------------
