import numpy as np
#from sklearn.preprocessing import MinMaxScaler

def order_batch(self, data, ind):
    d = []
    for i in data:
        d.append(i[ind])
    return(d)

def correct_sequence(data):
    for c,i in enumerate(data[:,0]):
        if c==0:
            lim = i
        else:
            if i<lim:
                lim = i
            else:
                return(False)
    return(True)

def data_loading(data, seq_len, n_signal):
    train_length = len(data)
    #data = np.vstack((data_train,data_test))
    # tabular = MinMaxScaler()
    # labels = MinMaxScaler()

    # Flip the data to make chronological data
    # data = data[::-1]
    tabular = data[::-1]

    # tabular = tabular.fit_transform(data)

    # Preprocess the dataset
    temp_data_train = []
    dec_data_train = []

    # Cut data by sequence length
    for i in range(0, len(tabular) - seq_len):
        if correct_sequence(tabular[i:i + seq_len]):
            _x = tabular[i:i + seq_len]
            pre_x = _x[:,1:n_signal]
            post_x = _x[:,n_signal+1:]
            # dec_data_train.append(labels.fit(_x[-1,n_signal]))
            dec_data_train.append(_x[-1,n_signal])

            _x = np.hstack((pre_x, post_x))
            temp_data_train.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data_train))
    data_train = []
    data_train_lab = []
    for i in range(len(temp_data_train)):
        data_train.append(temp_data_train[idx[i]])
        data_train_lab.append(dec_data_train[idx[i]])
    return([data_train,data_train_lab])
