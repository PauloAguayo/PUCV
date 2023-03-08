## Se importa aparte << mat73 >>, de lo contrario falla"
import mat73
import numpy as np
import pandas as pd
import random
from tqdm import tqdm


def finder(dataFrame,pattern):
    for t in range(len(dataFrame)):
        if float(pattern)==float(dataFrame[:][t][0]):
            return(dataFrame[t])


mat = mat73.loadmat('DATOSTJII_New7.mat')
w_length = 6

for key, value in mat.items():
    print(key)

print(mat['signals'])
print('Número de experimentos:',len(mat['array_shots']))


times_per_experiments = {}
descargas_out = []
for descarga in np.arange(len(mat['array_shots'])):
    min_times = []
    max_times = []
    try:
        for signal in np.arange(1,len(mat['signals'])+1): #### ARREGLAR EL 10
            times = mat['originalData'][descarga][signal][:,0]
            values = mat['originalData'][descarga][signal][:,1]
            min_times.append(np.trunc(100 * np.min(times)) / 100)
            max_times.append(np.trunc(100 * np.max(times)) / 100)
        min_time = np.max(np.array(min_times))
        max_time = np.min(np.array(max_times))
        times_per_experiments[str(mat['originalData'][descarga][0])]=[min_time,max_time]
    except:
        descargas_out.append(str(mat['originalData'][descarga][0]))

print('N° de descargas con alguna señal completa en nulo:',len(descargas_out))
print('N° de descargas:',len(times_per_experiments))

keys = list(times_per_experiments.keys())
random.shuffle(keys)
shuffled_times = dict()
for key in keys:
    shuffled_times.update({key: times_per_experiments[key]})


mins = []
for descarga in shuffled_times:
    windows = []
    for c,data in enumerate(finder(mat['originalData'],descarga)):
        if c!=0:
            data[:,0] = np.trunc(100 * data[:,0]) / 100
            df = pd.DataFrame(data, columns = ['time',mat['signals'][c-1]])
            df = df.loc[(df['time']>=float(shuffled_times[str(descarga)][0])) & (df['time']<=float(shuffled_times[str(descarga)][1]))]

            windows.append(len(df))
    mins.append(np.argmin(windows))

for d,(descarga,m) in tqdm(enumerate(zip(shuffled_times,mins))):
    ref_time = finder(mat['originalData'],descarga)[m+1][:,0]
    ref_time = np.trunc(100 * ref_time) / 100
    if d<int(len(shuffled_times)*0.8):
        for c,data in enumerate(finder(mat['originalData'],descarga)):
            if c!=0:
                activate = True
                data[:,0] = np.trunc(100 * data[:,0]) / 100
                df = pd.DataFrame(data, columns = ['time',mat['signals'][c-1]])
                df = df.loc[(df['time']>=float(shuffled_times[str(descarga)][0])) & (df['time']<=float(shuffled_times[str(descarga)][1]))]
                for tm in ref_time:
                    try:
                        df_f = df.loc[df['time']==float(tm)].iloc[:1]
                        if len(df_f)!=0 and activate:
                            activate = False
                            df_2f = df_f
                        elif len(df_f)!=0:
                            df_2f = pd.concat([df_2f, df_f], axis=0)
                    except:
                        continue
                df_2f.columns = ['time', mat['signals'][c-1]]
                if c == 1:
                    df2 = df_2f.sort_index()
                    df2 = df2.reset_index(drop=True)
                else:
                    df1 = df_2f.sort_index()
                    df1 = df1.reset_index(drop=True)
                    df1 = df1[mat['signals'][c-1]]
                    df2 = pd.concat([df2, df1], axis=1)
        if d == 0:
            df_train = df2
        else:
            df_train = pd.concat([df_train,df2],axis=0)
    else:
        for c,data in enumerate(finder(mat['originalData'],descarga)):
            if c!=0:
                activate = True
                data[:,0] = np.trunc(100 * data[:,0]) / 100
                df = pd.DataFrame(data, columns = ['time',mat['signals'][c-1]])
                df = df.loc[(df['time']>=float(shuffled_times[str(descarga)][0])) & (df['time']<=float(shuffled_times[str(descarga)][1]))]
                for tm in ref_time:
                    try:
                        df_f = df.loc[df['time']==float(tm)].iloc[:1]
                        if len(df_f)!=0 and activate:
                            activate = False
                            df_2f = df_f
                        elif len(df_f)!=0:
                            df_2f = pd.concat([df_2f, df_f], axis=0)
                    except:
                        continue
                df_2f.columns = ['time', mat['signals'][c-1]]
                if c == 1:
                    df2 = df_2f.sort_index()
                    df2 = df2.reset_index(drop=True)
                else:
                    df1 = df_2f.sort_index()
                    df1 = df1.reset_index(drop=True)
                    df1 = df1[mat['signals'][c-1]]
                    df2 = pd.concat([df2, df1], axis=1)
        if d == int(len(shuffled_times)*0.8):
            df_test = df2
        else:
            df_test = pd.concat([df_test,df2],axis=0)


df_train_no_na = df_train.dropna()
df_test_no_na = df_test.dropna()

print('Train dataset:', len(df_train))
print('Trest dataset:', len(df_test))
print('Train dataset (no na):', len(df_train_no_na))
print('Trest dataset (no na):', len(df_test_no_na))


df_train_no_na.to_csv('data_train_'+str(w_length)+'.csv',index=False)
df_test_no_na.to_csv('data_test_'+str(w_length)+'.csv',index=False)
