
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
import pandas as pd
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


seq_len = 24        # Timesteps
n_seq = 28          # Features
hidden_dim = 24     # Hidden units for generator (GRU & LSTM).
                    # Also decides output_units for generator
gamma = 1           # Used for discriminator loss
noise_dim = 32      # Used by generator as a starter dimension
dim = 128           # UNUSED
batch_size = 128
learning_rate = 5e-4
beta_1 = 0          # UNUSED
beta_2 = 1          # UNUSED
data_dim = 28       # UNUSED

# batch_size, lr, beta_1, beta_2, noise_dim, data_dim, layers_dim
gan_args = [batch_size, learning_rate, beta_1, beta_2, noise_dim, data_dim, dim]

file_path = "energydata_complete.csv"
energy_df = pd.read_csv(file_path)
energy_df['date'] = pd.to_datetime(energy_df['date'])
energy_df = energy_df.set_index('date').sort_index()

# Data transformations to be applied prior to be used with the synthesizer model
energy_data = real_data_loading(energy_df.values, seq_len=seq_len)
print(len(energy_data), energy_data[0].shape)

# Training
synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=1)
synth.train(energy_data, train_steps=500)
synth.save('synth_energy.pkl')
