#import tensorflow as tf
#gpu = len(tf.config.list_physical_devices('GPU'))>0
#print("GPU is", "available" if gpu else "NOT AVAILABLE")
import os
from support.utils import real_data_loading
import pandas as pd
from synthesizers.timeseries import TimeEncoder
from synthesizers import ModelParameters
from sklearn.decomposition import PCA#from sklearn.manifold import TSNE
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt


seq_len = 12        # Timesteps
n_seq = 8        # Features
hidden_dim = 8     # Hidden units for generator (GRU & LSTM).
                    # Also decides output_units for generator
gamma = 1           # Used for discriminator loss
noise_dim = 32      # Used by generator as a starter dimension
dim = 128           # UNUSED
batch_size = 128
learning_rate = 5e-4
beta_1 = 0          # UNUSED
beta_2 = 1          # UNUSED
data_dim = 28       # UNUSED

gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           noise_dim=noise_dim,
                           layers_dim=dim)



train_path = "data_train_24.csv"
train_df = pd.read_csv(train_path)


# Data transformations to be applied prior to be used with the synthesizer model
train_data = real_data_loading(train_df.values, seq_len=seq_len, n_signal=3)
print(len(train_data))#, train_data[0].shape)


synth = TimeEncoder(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=1, n_out=1)
synth.train(train_data, train_steps=500)

folders = os.listdir('models')
n = np.max([int(f.split('_')[1]) for f in folders])

try:
    folder = os.path.join('models','model_'+str(n+1))
    os.mkdir(folder)
except:
    folder = os.path.join('models','model_1')
    os.mkdir(folder)

print('Model saved in:',folder)
synth.save(os.path.join(folder,'synth_energy.pkl'))
