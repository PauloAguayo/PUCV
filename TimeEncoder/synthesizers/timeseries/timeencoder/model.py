from tqdm import tqdm, trange
import numpy as np

from tensorflow import function, GradientTape, sqrt, abs, reduce_mean, ones_like, zeros_like, convert_to_tensor, float32, config, reshape
from tensorflow import data as tfdata
from tensorflow import nn
from keras import (Model, Sequential, Input)
from keras.layers import (GRU, LSTM, Dense, Flatten)
from tensorflow.keras.optimizers import Adam
from keras.losses import (BinaryCrossentropy, MeanSquaredError)

from synthesizers.basemodel import BaseModel
#config.run_functions_eagerly(True)

def make_net(model, n_layers, hidden_units, output_units, net_type='GRU'):
    if net_type=='GRU':
        for i in range(n_layers):
            model.add(GRU(units=hidden_units,
                      return_sequences=True,
                      name=f'GRU_{i + 1}'))
    else:
        for i in range(n_layers):
            model.add(LSTM(units=hidden_units,
                      return_sequences=True,
                      name=f'LSTM_{i + 1}'))
    if output_units==1:
        model.add(Flatten())

        model.add(Dense(units=output_units,
                        activation='sigmoid',
                        name='OUT'))
        return(model)

    model.add(Dense(units=output_units,
                    activation='sigmoid',
                    name='OUT'))

    return model


class TimeEncoder(BaseModel):

    __MODEL__='TimeEncoder'

    def __init__(self, model_parameters, hidden_dim, seq_len, n_seq, gamma, n_out):
        self.seq_len=seq_len
        self.n_seq=n_seq
        self.n_out=n_out
        self.hidden_dim=hidden_dim
        self.gamma=gamma
        super().__init__(model_parameters)

    def define_encoder(self):
        self.recovery = Recovery(self.hidden_dim, self.n_out).build()
        self.embedder = Embedder(self.hidden_dim).build()

        X = Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RealData')

        #--------------------------------
        # Building the AutoEncoder
        #--------------------------------
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        self.autoencoder = Model(inputs=X, outputs=X_tilde)
        self.autoencoder.summary()
        self.embedder.summary()
        self.recovery.summary()


        # ----------------------------
        # Define the loss functions
        # ----------------------------
        self._mse=MeanSquaredError()
        self._bce=BinaryCrossentropy()


    @function
    def train_autoencoder(self, x, opt, y):
        x_real = reshape(y,[self.batch_size,1])
        with GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x_real, x_tilde)
            e_loss_0 = 10 * sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        y_true_mean, y_true_var = nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = nn.moments(x=y_pred, axes=[0])
        g_loss_mean = reduce_mean(abs(y_true_mean - y_pred_mean))
        g_loss_var = reduce_mean(abs(sqrt(y_true_var + 1e-6) - sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var

    def get_batch_data(self, data, n_windows):
        data = convert_to_tensor(data, dtype=float32)
        return iter(tfdata.Dataset.from_tensor_slices(data)
                                #.shuffle(buffer_size=n_windows)
                                .batch(self.batch_size).repeat())

    def _generate_noise(self):
        while True:
            yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.n_seq))

    def get_batch_noise(self):
        return iter(tfdata.Dataset.from_generator(self._generate_noise, output_types=float32)
                                .batch(self.batch_size)
                                .repeat())

    def order_batch(self, data, ind):
        d = []
        for i in data:
            d.append(i[ind])
        return(d)

    def train(self, data, train_steps):
        # Assemble the model
        self.define_encoder()

        ## Embedding network training
        autoencoder_opt = Adam(learning_rate=self.g_lr)
        for _ in tqdm(range(train_steps), desc='Emddeding network training'):
            X_ = next(self.get_batch_data(self.order_batch(data,0), n_windows=len(data)))
            y_ = next(self.get_batch_data(self.order_batch(data,1), n_windows=len(data)))
            step_e_loss_t0 = self.train_autoencoder(X_, autoencoder_opt,y_)

    def sample(self, n_samples):
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in trange(steps, desc='Synthetic data generation'):
            Z_ = next(self.get_batch_noise())
            records = self.generator(Z_)
            data.append(records)
        return np.array(np.vstack(data))

class Recovery(Model):
    def __init__(self, hidden_dim, n_out):
        self.hidden_dim=hidden_dim
        self.n_out=n_out
        return

    def build(self):
        recovery = Sequential(name='Recovery')
        recovery = make_net(recovery,
                            n_layers=3,
                            hidden_units=self.hidden_dim,
                            output_units=self.n_out)
        return recovery

class Embedder(Model):

    def __init__(self, hidden_dim):
        self.hidden_dim=hidden_dim
        return

    def build(self):
        embedder = Sequential(name='Embedder')
        embedder = make_net(embedder,
                            n_layers=3,
                            hidden_units=self.hidden_dim,
                            output_units=self.hidden_dim)
        return embedder
