print('--> Importing libraries...')
from utils import data_loading, model_path_pca
from Processes_pca import train
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", default=False, help="path to model")
parser.add_argument("-d", "--data_train", required=True, help="path to data train file")
parser.add_argument("-t", "--data_test", required=True, help="path to data test file")
# parser.add_argument("-i", "--input", default=0, type=str, help="path to optional input image file", required=True)
# parser.add_argument("-o", "--output", type=str, default="results/output.jpg", help="path and name to optional output image file")
parser.add_argument("-mt", "--model_type", type=str, default="GRU", help="GRU or LSTM")
parser.add_argument("-sl", "--sequence_length", type=int, default=24, help="sequence length for time interval")
parser.add_argument("-s", "--signal", type=str, default='Densidad2_', help="signal index as gt")
parser.add_argument("-hd", "--hidden_dim", type=int, default=8, help="hidden dim")
#parser.add_argument("-no", "--camera_height", type=float, default=2.5, help="z-coordinate for camera positioning")
parser.add_argument("-nl", "--number_of_layers", type=int, default=3, help="number of layers for model")
parser.add_argument("-b", "--batch_size", type=int, default=16, help="batch size")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate for ADAM")
parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
args = parser.parse_args()
vargs = vars(args)

seq_len = vargs["sequence_length"]
n_signal = 1
hidden_dim = vargs["hidden_dim"]
n_out = 1
n_layers = vargs["number_of_layers"]
batch_size = vargs["batch_size"]
learning_rate = vargs["learning_rate"] #0.0001
EPOCHS = vargs["epochs"] #100
model_type = vargs["model_type"]


dict_pca = {
    'ACTON275':0.813891,
    'BOL5':0.279392,
    'ECE7':-0.093833,
    'GR':0.099686,
    'GR2':-0.357565,
    'HALFAC3':0.182558,
    'IACCEL1':0.104093,
    'RX306':0.224596
}

print('--> Loading data...')
train_path = vargs["data_train"]
train_df_1 = pd.read_csv(train_path)

test_path = vargs["data_test"]
test_df_1 = pd.read_csv(test_path)

items_ordenados = sorted(dict_pca.items(), key=lambda x: abs(x[1]), reverse=True)
signals = ['time','Densidad2_']
for clave, valor in items_ordenados:
    signals.append(clave)
    print(signals)

    train_df = train_df_1[signals]
    test_df = test_df_1[signals]

    data_train = data_loading(train_df.values, seq_len=seq_len, n_signal=[n_signal])
    train_data = TensorDataset(torch.from_numpy(np.array(data_train[0])), torch.from_numpy(np.array(data_train[1])))
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)

    data_test = data_loading(test_df.values, seq_len=seq_len, n_signal=[n_signal])
    test_data = TensorDataset(torch.from_numpy(np.array(data_test[0])), torch.from_numpy(np.array(data_test[1])))
    test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print('--> torch device CUDA')
    else:
        device = torch.device("cpu")
        print('--> torch device CPU')

    path = os.path.join(model_path_pca(os.path.join('PCA','Analysis'),clave),'t_encoder.pth')

    time_model = train(train_loader, test_loader, learning_rate, hidden_dim, n_out, n_layers, batch_size, device, EPOCHS, 5000, 10, path, model_type, vargs["model"],seq_len)
