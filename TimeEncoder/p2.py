import torch
import torch.nn as nn
import numpy as np

# x_t = [[[-1.0493, -1.8555]]]#,
        # [[[ 0.5661, -0.5079]]]#,

        # [[-0.6644, -0.3921],
        #  [ 0.2802, -0.8306]]]

# x_t = [[[-0.8637,  0.2117],
x_t = [[-0.3897, -1.0932],

        # [ 0.4036, -0.8042]]],
         [ 1.4763,  0.0410]]
W_ih = [[ 0.2641, -0.0548],
        [ 0.2871, -0.0867]]
W_hh = [[-0.2182,  0.5277],
        [-0.0221, -0.4466]]
h_t_1 = [[-0.16061558, -0.31740706],
        [ 0.34496722, 0.10952165]]

# h = np.tanh(np.matmul(x_t,np.transpose(W_ih))+np.matmul(h_t_1,np.transpose(W_hh)))
_1 = np.matmul(x_t,np.transpose(W_ih))
_2 = np.matmul(h_t_1,np.transpose(W_hh))
h = _1+_2
print(np.tanh(h))
