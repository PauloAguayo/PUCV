from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
import pandas as pd
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt


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

synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=1)

synth.load('synth_energy.pkl')

synth_data = synth.sample(len(energy_data))

sample_size = 250
idx = np.random.permutation(len(energy_data))[:sample_size]

# Convert list to array, but taking only 250 random samples
# energy_data: (list(19711(ndarray(24, 28)))) -> real_sample: ndarray(250, 24, 28)
real_sample = np.asarray(energy_data)[idx]
synthetic_sample = np.asarray(synth_data)[idx]

# For the purpose of comparison we need the data to be 2-Dimensional.
# For that reason we are going to use only two components for both the PCA and TSNE.
# synth_data_reduced: {ndarray: (7000, 24)}
# energy_data_reduced: {ndarray: (7000, 24)}
synth_data_reduced = real_sample.reshape(-1, seq_len)
energy_data_reduced = np.asarray(synthetic_sample).reshape(-1,seq_len)

n_components = 2
pca = PCA(n_components=n_components)
tsne = TSNE(n_components=n_components, n_iter=300)

# The fit of the methods must be done only using the real sequential data
pca.fit(energy_data_reduced)

# pca_real: {DataFrame: (7000, 2)}
# pca_synth: {DataFrame: (7000, 2)}
pca_real = pd.DataFrame(pca.transform(energy_data_reduced))
pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

# data_reduced: {ndarray: (14000, 24)}
data_reduced = np.concatenate((energy_data_reduced, synth_data_reduced), axis=0)

# tsne_results: {DataFrame: (14000, 2)}
tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))


plt.scatter(pca_real,pca_synth)
plt.show()
