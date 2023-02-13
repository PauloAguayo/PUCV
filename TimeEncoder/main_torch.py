from utils import data_loading
from Processes import train, evaluate
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

seq_len = 24
n_signal = 3
hidden_dim = 8
n_out = 1
n_layers = 3
batch_size = 128
learning_rate = 0.001
EPOCHS = 2

train_path = "data_train_24.csv"
train_df = pd.read_csv(train_path)

test_path = "data_test_24.csv"
test_df = pd.read_csv(test_path)

data_train = data_loading(train_df.values, seq_len=seq_len, n_signal=n_signal)
#print(np.array(data_train[0]),np.array(data_train[1]))
train_data = TensorDataset(torch.from_numpy(np.array(data_train[0])), torch.from_numpy(np.array(data_train[1])))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

data_test = data_loading(test_df.values, seq_len=seq_len, n_signal=n_signal)
test_data = TensorDataset(torch.from_numpy(np.array(data_test[0])), torch.from_numpy(np.array(data_test[1])))
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


time_model = train(train_loader, learning_rate, hidden_dim, n_out, n_layers, batch_size, device, EPOCHS)

gru_outputs, targets, gru_sMAPE = evaluate(time_model, test_loader)
