import scipy.io as scio
import numpy as np
import tensorflow as tf


w = scio.loadmat("channel_weights.mat")['accuracy'][0, :]
result = np.empty([24, 1300, 26])
sum = np.zeros([1300, 26])
for i in range(24):
    result_dict = scio.loadmat('model_data/output_softmax'+str(i)+'.mat')
    result[i, :, :] = result_dict['output_softmax'+str(i)]
    sum[:, :] = sum[:, :] + w[i] * result[i, :, :]

data_test_Y = np.empty([1300])
for i in range(26):
    data_test_Y[50*i:50*i+50] = i
num_right = 0
for i in range(1300):
    if (np.argmax(sum[i, :]) == data_test_Y[i]):
        num_right += 1
print(num_right/1300)