import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch


class Logs(object):
    def __init__(self, path, learn_rate, hidden_dim, n_out, n_layers, batch_size, EPOCHS, model_type, true_model, seq_len, best_loss_train, best_loss_test, model_embedder, model_estimator):
        self.path = path
        self.learn_rate = learn_rate
        self.hidden_dim = hidden_dim
        self.n_out = n_out
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.EPOCHS = EPOCHS
        self.model_type = model_type
        self.true_model = true_model
        self.seq_len = seq_len
        self.best_loss_train = best_loss_train
        self.best_loss_test = best_loss_test
        self.model_emb = model_embedder
        self.model_est = model_estimator

    def losses(self, train_lines, test_lines):
        for x,i in enumerate(['_train_losses.txt','_test_losses.txt']):
            path_ = self.path.split('.')[0] + i
            with open(path_, 'w') as f:
                if x==0:
                    lines = train_lines
                else:
                    lines = test_lines

                for c,line in enumerate(lines):
                    if c==len(lines)-1:
                        f.write(str(line))
                    else:
                        f.write(str(line))
                        f.write('\n')

    def best_model(self, train_value, test_value, model_est, model_emb):

        def details(train_score, test_score):
            path_ = r"{}".format(self.path)
            lr = 'learning rate = '+str(self.learn_rate)
            hd = 'hidden dim = '+str(self.hidden_dim)
            no = 'n neurons output = '+str(self.n_out)
            nl = 'n layers = '+str(self.n_layers)
            bs = 'batch size = '+str(self.batch_size)
            e = 'epochs = '+str(self.EPOCHS)
            mt = 'model type = '+str(self.model_type)
            tm = 'previous model = '+str(self.true_model)
            bte = 'best train error = '+str(train_score)
            bt = 'best test error = '+str(test_score)
            sl = 'sequence length = '+str(self.seq_len)
            lines = [lr, hd, no, nl, bs, e, mt, tm, bte, bt, sl]
            new_path = path_.split("\\")[:-1]
            with open(r"{}".format(os.path.join(*new_path,'model_details.txt')), 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')

        if train_value<self.best_loss_train:
            self.best_loss_train = train_value
            self.model_emb = model_emb
            save_path_emb = self.path.split('.')[0]+'_embedder_train_best.'+self.path.split('.')[1]
            save_path_est = self.path.split('.')[0]+'_estimator_train_best.'+self.path.split('.')[1]
            torch.save(self.model_emb.state_dict(), save_path_emb)
            torch.save(self.model_est.state_dict(), save_path_est)
        if test_value<self.best_loss_test:
            self.best_loss_test = test_value
            self.model_est = model_est
            save_path_emb = self.path.split('.')[0]+'_embedder_test_best.'+self.path.split('.')[1]
            save_path_est = self.path.split('.')[0]+'_estimator_test_best.'+self.path.split('.')[1]
            torch.save(self.model_emb.state_dict(), save_path_emb)
            torch.save(self.model_est.state_dict(), save_path_est)

        details(self.best_loss_train, self.best_loss_test)
        return(self.best_loss_train, self.best_loss_test)


def model_path(path):
    if not os.listdir(path): # empty directory
        os.mkdir(os.path.join(path,'model_1'))
        return(os.path.join(path,'model_1'))
    else:
        for i in os.listdir(path):
            new_path = i.split('_')[0]+'_'+str(int(i.split('_')[1])+1)
            if not os.path.exists(os.path.join(path,new_path)):
                os.mkdir(os.path.join(path,new_path))
                return(os.path.join(path,new_path))

def order_batch(self, data, ind):
    d = []
    for i in data:
        d.append(i[ind])
    return(d)

def correct_chronological_sequence(data):
    for c,i in enumerate(data[:,0]):
        if c==0:
            lim = i
        else:
            if i<lim:
                lim = i
            else:
                return(False)
    return(True)

def correct_sequence(data):
    for c,i in enumerate(data[:,0]):
        if c==0:
            lim = i
        else:
            if i>=lim:
                lim = i
            else:
                return(False)
    return(True)

def data_loading(data, seq_len, n_signal):
    train_length = len(data)
    # tab = MinMaxScaler()

    # Flip the data to make chronological data
    # data = data[::-1]
    # tabular = data[::-1]

    tabular = data
    # tabular = tab.fit_transform(data)

    # Preprocess the dataset
    train_data = []
    labels_data = []

    # Cut data by sequence length
    for i in range(0, len(tabular) - seq_len):
        # if correct_chronological_sequence(tabular[i:i + seq_len]):
        if correct_sequence(tabular[i:i + seq_len]):
            _x = tabular[i:i + seq_len]
            pre_x = _x[:,1:n_signal]
            post_x = _x[:,n_signal+1:]
            labels_data.append(_x[-1,n_signal])

            _x = np.hstack((pre_x, post_x))
            train_data.append(_x)


    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(train_data))
    data_train = []
    data_train_lab = []
    for i in range(len(train_data)):
        data_train.append(train_data[idx[i]])
        data_train_lab.append(labels_data[idx[i]])
    return([data_train,data_train_lab])
