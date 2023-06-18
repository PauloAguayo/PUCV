## Se importa aparte << mat73 >>, de lo contrario falla"
import mat73
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


def finder(dataFrame,pattern):
    for t in range(len(dataFrame)):
        if float(pattern)==float(dataFrame[:][t][0]):
            return(dataFrame[t])


mat = mat73.loadmat('DATOSTJII_New7.mat')

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

with open(r"{}".format('descargas_na.txt'), 'w') as f:
    for line in descargas_out:
        f.write(line)
        f.write('\n')

keys = list(times_per_experiments.keys())
random.shuffle(keys)
shuffled_times = dict()
for key in keys:
    shuffled_times.update({key: times_per_experiments[key]})

file1 = open(r"{}".format('descargas_no_ok.txt'), 'r')
Lines = file1.readlines()

count__ = []
# Strips the newline character
for line in Lines:
    count__.append(line.strip())


mins = []
señales_no_ok = []
for cc,descarga in enumerate(shuffled_times):
    windows = []
    si = True
    for c,data in enumerate(finder(mat['originalData'],descarga)):
        if c!=0:
            data[:,0] = np.trunc(100 * data[:,0]) / 100
            df = pd.DataFrame(data, columns = ['time',mat['signals'][c-1]])
            df = df.loc[(df['time']>=float(shuffled_times[str(descarga)][0])) & (df['time']<=float(shuffled_times[str(descarga)][1]))]

            windows.append(len(df))
            if c==3:
                if str(descarga) in count__:
                    señales_no_ok.append(descarga)
                    si = False
                # df.loc[df[mat['signals'][c-1]] <= 0.0, mat['signals'][c-1]] = np.random.uniform(0.001,0.009,1)[0]
                # plt.plot(df['time'].values,df[mat['signals'][c-1]].values)
                # plt.plot(df['time'].values,np.zeros(len(df[mat['signals'][c-1]].values)))
                # plt.savefig('books_read.png')
                # img = cv2.imread("books_read.png")
                # while(True):
                    # cv2.imshow('Handcrafted image',img)
                    # k = cv2.waitKey(1) & 0xFF
                    # if k == ord('a'):
                        # señales_no_ok.append(descarga)
                        # cv2.destroyAllWindows()
                        # plt.clf()
                        # print(descarga)
                        # si = False
                        # break
                    # elif k== ord('s'):
                    #     cv2.destroyAllWindows()
                    #     plt.clf()
                    #     break
    if si:
        mins.append(np.argmin(windows))

# with open(r"{}".format('descargas_no_ok.txt'), 'w') as f:
#     for line in señales_no_ok:
#         f.write(line)
#         f.write('\n')

descargas_test = []
descargas_train = []

print(len(shuffled_times))
for key in señales_no_ok:
    shuffled_times.pop(key,None)
print(len(shuffled_times))

for d,(descarga,m) in tqdm(enumerate(zip(shuffled_times,mins))):
    ref_time = finder(mat['originalData'],descarga)[m+1][:,0]
    ref_time = np.trunc(100 * ref_time) / 100
    if d<int(len(shuffled_times)*0.8):
        descargas_train.append(descarga)
        delete = []
        for c,data in enumerate(finder(mat['originalData'],descarga)):
            if c!=0:
                activate = True
                data[:,0] = np.trunc(100 * data[:,0]) / 100
                df = pd.DataFrame(data, columns = ['time',mat['signals'][c-1]])
                df = df.loc[(df['time']>=float(shuffled_times[str(descarga)][0])) & (df['time']<=float(shuffled_times[str(descarga)][1]))]
                for tm in ref_time:
                        df_f = df.loc[df['time']==float(tm)].iloc[:1]
                        if len(df_f)!=0 and activate:
                            activate = False
                            df_2f_raw = df_f
                            df_2f_pr = df_f
                        elif len(df_f)!=0:
                            df_2f_raw = pd.concat([df_2f_raw, df_f], axis=0)
                            df_2f_pr = pd.concat([df_2f_pr, df_f], axis=0)
                        elif len(df_f)==0 and activate:
                            activate = False
                            ghost = {'time': [float(tm)], mat['signals'][c-1]: [0.0]}
                            df_ghost = pd.DataFrame(data=ghost)
                            delete.append(tm)
                            df_2f_raw = df_ghost
                            df_2f_pr = df_ghost
                        elif len(df_f)==0:
                            ghost = {'time': [float(tm)], mat['signals'][c-1]: [0.0]}
                            df_ghost = pd.DataFrame(data=ghost)
                            delete.append(tm)
                            df_2f_raw = pd.concat([df_2f_raw, df_ghost], axis=0)
                            df_2f_pr = pd.concat([df_2f_pr, df_ghost], axis=0)
                    # except:
                    #     continue
                df_2f_raw.columns = ['time', mat['signals'][c-1]]
                df_2f_pr.columns = ['time', mat['signals'][c-1]]
                if c==3:
                    df_2f_pr.loc[df_2f_pr[mat['signals'][c-1]] <= 0.0, mat['signals'][c-1]] = np.random.uniform(0.001,0.009,1)[0]

                if c == 1:
                    df2_raw = df_2f_raw.sort_index()
                    df2_raw = df2_raw.reset_index(drop=True)

                    df2_pr = df_2f_pr.sort_index()
                    df2_pr = df2_pr.reset_index(drop=True)
                else:
                    df1_raw = df_2f_raw.sort_index()
                    df1_raw = df1_raw.reset_index(drop=True)
                    df1_raw = df1_raw[mat['signals'][c-1]]
                    df2_raw = pd.concat([df2_raw, df1_raw], axis=1)

                    df1_pr = df_2f_pr.sort_index()
                    df1_pr = df1_pr.reset_index(drop=True)
                    df1_pr = df1_pr[mat['signals'][c-1]]
                    df2_pr = pd.concat([df2_pr, df1_pr], axis=1)
        delete = np.unique(delete)
        print(len(df2_pr))
        for supr in delete:
            df2_raw = df2_raw.loc[df2_raw['time']!=float(supr)]
            df2_pr = df2_pr.loc[df2_pr['time']!=float(supr)]

        print(len(df2_pr))
        if d == 0:
            df_train_raw = df2_raw
            df_train_pr = df2_pr
        else:
            df_train_raw = pd.concat([df_train_raw,df2_raw],axis=0)
            df_train_pr = pd.concat([df_train_pr,df2_pr],axis=0)
    else:
        descargas_test.append(descarga)
        delete = []
        for c,data in enumerate(finder(mat['originalData'],descarga)):
            if c!=0:
                activate = True
                data[:,0] = np.trunc(100 * data[:,0]) / 100
                df = pd.DataFrame(data, columns = ['time',mat['signals'][c-1]])
                df = df.loc[(df['time']>=float(shuffled_times[str(descarga)][0])) & (df['time']<=float(shuffled_times[str(descarga)][1]))]
                for tm in ref_time:
                        df_f = df.loc[df['time']==float(tm)].iloc[:1]
                        if len(df_f)!=0 and activate:
                            activate = False
                            df_2f_raw = df_f
                            df_2f_pr = df_f
                        elif len(df_f)!=0:
                            df_2f_raw = pd.concat([df_2f_raw, df_f], axis=0)
                            df_2f_pr = pd.concat([df_2f_pr, df_f], axis=0)
                        elif len(df_f)==0 and activate:
                            activate = False
                            ghost = {'time': [float(tm)], mat['signals'][c-1]: [0.0]}
                            df_ghost = pd.DataFrame(data=ghost)
                            delete.append(tm)
                            df_2f_raw = df_ghost
                            df_2f_pr = df_ghost
                        elif len(df_f)==0:
                            ghost = {'time': [float(tm)], mat['signals'][c-1]: [0.0]}
                            df_ghost = pd.DataFrame(data=ghost)
                            delete.append(tm)
                            df_2f_raw = pd.concat([df_2f_raw, df_ghost], axis=0)
                            df_2f_pr = pd.concat([df_2f_pr, df_ghost], axis=0)
                    # except:
                    #     continue
                df_2f_raw.columns = ['time', mat['signals'][c-1]]
                df_2f_pr.columns = ['time', mat['signals'][c-1]]
                if c==3:
                    df_2f_pr.loc[df_2f_pr[mat['signals'][c-1]] <= 0.0, mat['signals'][c-1]] = np.random.uniform(0.001,0.009,1)[0]
                if c == 1:
                    df2_raw = df_2f_raw.sort_index()
                    df2_raw = df2_raw.reset_index(drop=True)

                    df2_pr = df_2f_pr.sort_index()
                    df2_pr = df2_pr.reset_index(drop=True)
                else:
                    df1_raw = df_2f_raw.sort_index()
                    df1_raw = df1_raw.reset_index(drop=True)
                    df1_raw = df1_raw[mat['signals'][c-1]]
                    df2_raw = pd.concat([df2_raw, df1_raw], axis=1)

                    df1_pr = df_2f_pr.sort_index()
                    df1_pr = df1_pr.reset_index(drop=True)
                    df1_pr = df1_pr[mat['signals'][c-1]]
                    df2_pr = pd.concat([df2_pr, df1_pr], axis=1)
        delete = np.unique(delete)
        print(len(df2_pr))
        for supr in delete:
            df2_raw = df2_raw.loc[df2_raw['time']!=float(supr)]
            df2_pr = df2_pr.loc[df2_pr['time']!=float(supr)]

        print(len(df2_pr))
        if d == int(len(shuffled_times)*0.8):
        # if d == 0:
            df_test_raw = df2_raw
            df_test_pr = df2_pr
        else:
            df_test_raw = pd.concat([df_test_raw,df2_raw],axis=0)
            df_test_pr = pd.concat([df_test_pr,df2_pr],axis=0)


with open(r"{}".format('descargas_train.txt'), 'w') as f:
    for line in descargas_train:
        f.write(line)
        f.write('\n')

with open(r"{}".format('descargas_test.txt'), 'w') as f:
    for line in descargas_test:
        f.write(line)
        f.write('\n')

df_train_no_na_raw = df_train_raw.dropna()
df_test_no_na_raw = df_test_raw.dropna()

df_train_no_na_pr = df_train_pr.dropna()
df_test_no_na_pr = df_test_pr.dropna()

print('Train dataset:', len(df_train_raw))
print('Trest dataset:', len(df_test_raw))
print('Train dataset (no na):', len(df_train_no_na_raw))
print('Trest dataset (no na):', len(df_test_no_na_raw))


df_train_no_na_raw.to_csv('data_train_raw_def.csv',index=False)
df_test_no_na_raw.to_csv('data_test_raw_def.csv',index=False)

df_train_no_na_pr.to_csv('data_train_def.csv',index=False)
df_test_no_na_pr.to_csv('data_test_def.csv',index=False)
