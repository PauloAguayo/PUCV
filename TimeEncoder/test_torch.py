print('--> Importing libraries...')
from utils import data_loading, model_path
from Processes import evaluate
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


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

seq_len = vargs["sequence_length"] #24
n_signal = vargs["signal_number"] #3
hidden_dim = vargs["hidden_dim"] #8
# n_out = 1
# n_layers = vargs["number_of_layers"] #3
batch_size = vargs["batch_size"] #16
# learning_rate = vargs["learning_rate"] #0.0001
# EPOCHS = vargs["epochs"] #100
# model_type = vargs["model_type"]


print('--> Loading data...')
test_path = vargs["data_test"] #"data_train_24.csv"
test_df = pd.read_csv(test_path)

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
