import torch
from Models import Embedder, Estimator
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


file = pd.read_csv('data_test_def.csv')
# file_referencia = pd.read_csv('data_train_def.csv')
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

input_dim = hidden_dim = 8
n_layers = 3
batch_size = 1
device = torch.device("cuda")
model_type = 'GRU'
bidirectional = False
n_out = 1

model_embedder = Embedder(input_dim, hidden_dim, n_layers, batch_size, device, model_type, bidirectional)
model_estimator = Estimator(input_dim, hidden_dim, n_out, n_layers, batch_size, device, model_type, bidirectional)

model_embedder.load_state_dict(torch.load('models/model_91/t_encoder_embedder_best.pth'))
model_estimator.load_state_dict(torch.load('models/model_91/t_encoder_estimator_best.pth'))
model_embedder.eval()
model_estimator.eval()

tabular = diccionario['10114.0'].values
seq_len = 6
n_signal = 3
train_data = []
labels_data = []
time = []

# tab = MinMaxScaler()

# Flip the data to make chronological data
# data = data[::-1]
#tabular = data[::-1]


# tabular = data
# tabular_ref = tab.fit(df.iloc[:,0].values.reshape(-1,1))    tab.fit_transform(file_referencia.values)

for i in range(0, len(tabular) - seq_len):
    _x = tabular[i:i + seq_len]
    time.append(str(_x[-1,0]))
    pre_x = _x[:,1:n_signal]
    post_x = _x[:,n_signal+1:]
    labels_data.append(_x[-1,n_signal])

    _x = np.hstack((pre_x, post_x))
    train_data.append(_x)


train_data = TensorDataset(torch.from_numpy(np.array(train_data)), torch.from_numpy(np.array(labels_data)))
train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=False, shuffle=False)
estimaciones = []
outputs = []

h_e = model_embedder.init_hidden()
h_r = model_estimator.init_hidden()

print(len(train_loader))

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
    h_e = torch.permute(h_e, (1, 0, 2))

    out_r, h_r = model_estimator(out_e.to(device).float(), h_r)
    h_r = torch.permute(h_r, (1, 0, 2))
    estimaciones.append(out_r[0][0].cpu().detach().numpy().reshape(-1))
    outputs.append(label.numpy().reshape(-1))

sMAPE = 0
count = 0
for i in range(len(outputs)):
    for y,y__ in zip(outputs[i],estimaciones[i]):
        count+=1
        sMAPE += abs(y__-y)/(abs(y)+abs(y__))/2
print("sMAPE: {}%".format(sMAPE/count*100))

with open('estimaciones.txt', 'w') as f:
    for line,ti in zip(estimaciones,time):
        f.write(str(line))
        f.write(',')
        f.write(ti)
        f.write('\n')
