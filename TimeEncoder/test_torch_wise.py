print('--> Importing libraries...')
from utils import data_loading, model_path
from Processes import evaluate
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
# import argparse
# import matplotlib.pyplot as plt
import mat73

#
def details(path, hidden_dim, batch_size, seq_len, score ):
    path_ = r"{}".format(path)
    hd = 'hidden dim = '+str(hidden_dim)
    bs = 'batch size = '+str(batch_size)
    sl = 'sequence length = '+str(seq_len)
    smape = 'sMAPE = '+str(score)
    lines = [hd, bs, sl, smape]
    new_path = path_.split("\\")[:-1]
    with open(r"{}".format(os.path.join(*new_path,'results.txt')), 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
#
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", required=True, help="path to model")
parser.add_argument("-d", "--data_test", required=True, help="path to data train file")
# parser.add_argument("-i", "--input", default=0, type=str, help="path to optional input image file", required=True)
# parser.add_argument("-o", "--output", type=str, default="results/output.jpg", help="path and name to optional output image file")
# parser.add_argument("-mt", "--model_type", type=str, default="GRU", help="GRU or LSTM")
parser.add_argument("-sl", "--sequence_length", type=int, default=24, help="sequence length for time interval")
parser.add_argument("-sn", "--signal_number", type=int, default=3, help="signal index as gt")
parser.add_argument("-hd", "--hidden_dim", type=int, default=8, help="hidden dim")
#parser.add_argument("-no", "--camera_height", type=float, default=2.5, help="z-coordinate for camera positioning")
# parser.add_argument("-nl", "--number_of_layers", type=int, default=3, help="number of layers for model")
parser.add_argument("-b", "--batch_size", type=int, default=16, help="batch size")
# parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate for ADAM")
# parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
args = parser.parse_args()
vargs = vars(args)
#
seq_len = vargs["sequence_length"] #24
n_signal = vargs["signal_number"] #3
hidden_dim = vargs["hidden_dim"] #8
# n_out = 1
# n_layers = vargs["number_of_layers"] #3
# batch_size = vargs["batch_size"] #16
# learning_rate = vargs["learning_rate"] #0.0001
# EPOCHS = vargs["epochs"] #100
# model_type = vargs["model_type"]
#
#
print('--> Loading data...')
test_path = vargs["data_test"] #"data_train_24.csv"
test_df = pd.read_csv(test_path)

# n_signals = 9
# mat = mat73.loadmat('DATOSTJII_New7.mat')
#
# for i in range(1,n_signals+1):
#     print(i)
#     d = {'time':mat['originalData'][0][i][:,0], 'values':mat['originalData'][0][i][:,1]}
#     mat_2 = pd.DataFrame(data=d)
#     mat_2['time'] = np.trunc(100 * mat_2['time']) / 100
#
#     mat_2 = mat_2.loc[(mat_2['time']>=1000.0) & (mat_2['time']<=1386.64)]
#     mat_2.drop_duplicates(['time'], inplace=True)
#     print(len(mat_2))

#
# for i in range(0, len(mat) - 12):
#     _x = tabular[i:i + seq_len]
#     pre_x = _x[:,1:n_signal]
#     post_x = _x[:,n_signal+1:]
#     # dec_data_train.append(labels.fit(_x[-1,n_signal]))
#     dec_data_train.append(_x[-1,n_signal])
#
#     _x = np.hstack((pre_x, post_x))
#     temp_data_train.append(_x)
#
#
# mins = []
# for descarga in shuffled_times:
#     windows = []
#     for c,data in enumerate(finder(mat['originalData'],descarga)):
#         if c!=0:
#             data[:,0] = np.trunc(100 * data[:,0]) / 100
#             df = pd.DataFrame(data, columns = ['time',mat['signals'][c-1]])
#             df = df.loc[(df['time']>=float(shuffled_times[str(descarga)][0])) & (df['time']<=float(shuffled_times[str(descarga)][1]))]
#
#             windows.append(len(df))
#     mins.append(np.argmin(windows))
#
# for d,(descarga,m) in tqdm(enumerate(zip(shuffled_times,mins))):
#     ref_time = finder(mat['originalData'],descarga)[m+1][:,0]
#     ref_time = np.trunc(100 * ref_time) / 100
#     if d<int(len(shuffled_times)*0.8):
#         for c,data in enumerate(finder(mat['originalData'],descarga)):
#             if c!=0:
#                 activate = True
#                 data[:,0] = np.trunc(100 * data[:,0]) / 100
#                 df = pd.DataFrame(data, columns = ['time',mat['signals'][c-1]])
#                 df = df.loc[(df['time']>=float(shuffled_times[str(descarga)][0])) & (df['time']<=float(shuffled_times[str(descarga)][1]))]
#                 for tm in ref_time:
#                     try:
#                         df_f = df.loc[df['time']==float(tm)].iloc[:1]
#                         if len(df_f)!=0 and activate:
#                             activate = False
#                             df_2f = df_f
#                         elif len(df_f)!=0:
#                             df_2f = pd.concat([df_2f, df_f], axis=0)
#                     except:
#                         continue
#                 df_2f.columns = ['time', mat['signals'][c-1]]
#                 if c == 1:
#                     df2 = df_2f.sort_index()
#                     df2 = df2.reset_index(drop=True)
#                 else:
#                     df1 = df_2f.sort_index()
#                     df1 = df1.reset_index(drop=True)
#                     df1 = df1[mat['signals'][c-1]]
#                     df2 = pd.concat([df2, df1], axis=1)



data_test = data_loading(test_df.values, seq_len=seq_len, n_signal=n_signal)
test_data = TensorDataset(torch.from_numpy(np.array(data_test[0])), torch.from_numpy(np.array(data_test[1])))
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print('--> torch device CUDA')
else:
    device = torch.device("cpu")
    print('--> torch device CPU')

time_model = torch.load(vargs["model"])

outputs, targets, sMAPE = evaluate(time_model, test_loader, device)

details(vargs["model"],hidden_dim, batch_size, seq_len,sMAPE)

plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
plt.plot(outputs[0][:], "-o", color="g", label="Predicted")
plt.plot(targets[0][:], color="b", label="Actual")
plt.ylabel('Densidad2_')
plt.legend()

plt.subplot(2,2,2)
plt.plot(outputs[8][-50:], "-o", color="g", label="Predicted")
plt.plot(targets[8][-50:], color="b", label="Actual")
plt.ylabel('Densidad2_')
plt.legend()

plt.subplot(2,2,3)
plt.plot(outputs[4][:50], "-o", color="g", label="Predicted")
plt.plot(targets[4][:50], color="b", label="Actual")
plt.ylabel('Densidad2_')
plt.legend()

plt.subplot(2,2,4)
plt.plot(outputs[6][:100], "-o", color="g", label="Predicted")
plt.plot(targets[6][:100], color="b", label="Actual")
plt.ylabel('Densidad2_')
plt.legend()
new_path = vargs["model"].split("\\")[:-1]
plt.savefig(r"{}".format(os.path.join(*new_path,'plot.png')))
plt.show()
