import torch
from Models_complex import Embedder, Estimator
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils import exponential_smoothing
import os


file = pd.read_csv('data_test_def.csv')
descargas = open('descargas_test.txt','r')

diccionario = {}
max_init = 0
max_fin = 0
for d in descargas:
    file_temporal = file.iloc[max_init:]
    time = file_temporal['time'].tolist()
    for c,t in enumerate(time):
        if c>0:
            if time[c-1]<=t:
                max_fin+=1
            else:
                break
    diccionario[d[:-1]] = file.iloc[max_init:max_fin+max_init+1]
    max_init = max_fin+max_init+1
    max_fin = 0

input_dim = 8
hidden_dim = 36
n_layers = 3
batch_size = 1
device = torch.device("cuda")
model_type = 'GRU'
n_out = 1
seq_len = 9
n_signal = 3

model_embedder = Embedder(input_dim, hidden_dim, n_layers, batch_size, device, model_type)
model_estimator = Estimator(input_dim, hidden_dim, n_out, n_layers, batch_size, device, model_type, seq_len)

model_embedder.load_state_dict(torch.load('/home/paulo/Documents/modelos_tesis/GRU/t_encoder_embedder_test_best.pth'))# 39
model_estimator.load_state_dict(torch.load('/home/paulo/Documents/modelos_tesis/GRU/t_encoder_estimator_test_best.pth')) # 39
model_embedder.eval()
model_estimator.eval()

for tab in diccionario:
    print(tab)
    tabular = diccionario[tab].values
    train_data = []
    labels_data = []
    time = []
    for i in range(0, len(tabular) - seq_len):
        _x = tabular[i:i + seq_len]
        time.append(str(_x[-1,0]))
        if i == 0:
            active = True
            past = 0 #np.zeros(np.shape(tabular[0])[0])

        _x, past = exponential_smoothing(_x,active,past)
        active = False

        pre_x = _x[:,1:n_signal]
        post_x = _x[:,n_signal+1:]
        labels_data.append(_x[-1,n_signal])

        _x = np.hstack((pre_x, post_x))
        train_data.append(_x)


    train_data = TensorDataset(torch.from_numpy(np.array(train_data)), torch.from_numpy(np.array(labels_data)))
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=False, shuffle=False)

    outputs = []
    estimaciones = []

    for batch, label in train_loader:
        h_e = model_embedder.init_hidden()
        h_r = model_estimator.init_hidden()
        if torch.cuda.is_available():
            model_embedder.cuda()
            model_estimator.cuda()
            batch = batch.cuda()
        if model_type == "GRU":
            h_e = h_e.data
            h_r = h_r.data
        else:
            h_e = tuple([e.data for e in h_e])
            h_r = tuple([e.data for e in h_r])

        out_e, h_e = model_embedder(batch.to(device).float(), h_e)
        #h_e = torch.permute(h_e, (1, 0, 2))

        out_r, h_r = model_estimator(out_e.to(device).float(), h_r)
        #h_r = torch.permute(h_r, (1, 0, 2))
        estimaciones.append(out_r[0][0].cpu().detach().numpy().reshape(-1)[0])
        outputs.append(label.numpy().reshape(-1)[0])


    with open(os.path.join('data','results_test',str(tab)+'.txt'), 'w') as f:
        for outs,line,ti in zip(outputs,estimaciones,time):
            f.write(ti)
            f.write(',')
            f.write(str(outs))
            f.write(',')
            f.write(str(line))
            f.write('\n')


# sMAPE = 0
# count = 0
# for i in range(len(outputs)):
#     for y,y__ in zip(outputs[i],estimaciones[i]):
#         count+=1
#         sMAPE += abs(y__-y)/(abs(y)+abs(y__))/2
# print("sMAPE: {}%".format(sMAPE/count*100))
#
# with open('estimaciones.txt', 'w') as f:
#     for line,ti in zip(estimaciones,time):
#         f.write(str(line))
#         f.write(',')
#         f.write(ti)
#         f.write('\n')
