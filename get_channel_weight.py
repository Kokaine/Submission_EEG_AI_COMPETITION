import numpy as np
import scipy.io as scio
from KNN import KNN

raw_data = scio.loadmat('divided_data_PCA.mat')
data_train = raw_data['data_train']
label_train = raw_data['label_train']
data_test = raw_data['data_test']
label_test = raw_data['label_test']

k = 20
accuracy = np.empty([24])
for i in range(24):
    acc, totalNum = KNN(data_train[:, :, i], label_train[0, :], data_test[:, :, i], label_test[0, :], k = k)
    print('K = '+str(k)+' channel'+str(i)+'acc:', sum(acc)/sum(totalNum))
    accuracy[i] = sum(acc)/sum(totalNum)

scio.savemat('channel_weights.mat', mdict={'accuracy' : accuracy})
