import numpy as np
import scipy.io as scio

# calculate the distance between x(1xn) and y(1xn)
def dis(x, y):
    return sum((x-y)**2)

# calculate confusion matrix of testData
def confusion_mat(testLabel, predict, num_label = 26):
    # conf_mat[i, j] represents the number of original label i predicted to label j
    conf_mat = np.zeros([num_label, num_label])
    p = np.size(testLabel)
    for i in range(p):
        conf_mat[int(testLabel[i]), int(predict[i])] += 1
    conf_mat /= 30
    return conf_mat



# K Nearest Neighbors
# GT dataMat mxn; m data; n feature
# GT dataLabel mx1; label of each data
# testData pxn
# testLabel px1
def KNN(dataMat, dataLabel, testData, testLabel, num_label = 26, k = 20):
    m, n = np.shape(dataMat)
    p, n = np.shape(testData)
    predict = np.empty([p])
    acc = np.zeros([num_label])
    totalNum = np.zeros([num_label])
    for i in range(p):
        list = []
        for j in range(m):
            list.append(dis(testData[i, :], dataMat[j, :]))
        neighborsInd = np.argsort(list)[:k]
        neighbors = np.zeros([num_label])
        for j in range(k):
            neighbors[int(dataLabel[neighborsInd[j]])] += 1
        predict[i] = np.argmax(neighbors)
        totalNum[int(testLabel[i])] += 1
        if (predict[i] == testLabel[i]):
            acc[int(testLabel[i])] += 1

    # return confusion_mat(testLabel, predict, num_label)
    return acc, totalNum



