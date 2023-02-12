import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gc
import sys
# Method implemented here: https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py
# Originally used in TimeGAN research

def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz

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


def real_data_loading(data: np.array, seq_len, n_signal):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    # Flip the data to make chronological data
    ori_data = data[::-1]
    # Normalize the data
    scaler = MinMaxScaler().fit(ori_data)
    ori_data = scaler.transform(ori_data)

    # Preprocess the dataset
    temp_data = []
    dec_data = []

    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        # if i >= (len(ori_data) - seq_len)*1/4:
        #     break
        if correct_sequence(ori_data[i:i + seq_len]):
            _x = ori_data[i:i + seq_len]
            pre_x = _x[:,1:n_signal]
            post_x = _x[:,n_signal+1:]
            dec_data.append(_x[-1,n_signal])

            _x = np.hstack((pre_x, post_x))
            temp_data.append(_x)

            # if get_obj_size({'x':temp_data, 'y':dec_data})>3000000000:
            #     print(i)
            #     break
    print(get_obj_size({'x':temp_data, 'y':dec_data}))
    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append([temp_data[idx[i]],dec_data[idx[i]]])
    return data
