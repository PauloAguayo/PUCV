import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import function, GradientTape, sqrt, abs, reduce_mean, ones_like, zeros_like, convert_to_tensor,float32

from keras.losses import (BinaryCrossentropy, MeanSquaredError)

y_true = [[0., 1.], [0., 0.]]
y_pred = [[1., 1.], [1., 0.]]
# Using 'auto'/'sum_over_batch_size' reduction type.
mse = MeanSquaredError()
print(10 * sqrt(mse(y_true, y_pred)))

# print(y_true+[])
