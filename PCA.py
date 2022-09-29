import numpy as np
import scipy.io as scio

# Principal components analyses
# m number of data
# n number of feature
def PCA(dataMat, k = 999999):
    meanValues = np.mean(dataMat,axis=0) # 竖着求平均值，数据格式是m×n
    meanRemoved = dataMat - meanValues  # 0均值化  m×n维
    covMat = np.cov(meanRemoved,rowvar=0)  # 每一列作为一个独立变量求协方差  n×n维
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) # 求特征值和特征向量  eigVects是n×n维
    totalVar = sum(eigVals)
    eigValInd = np.argsort(-eigVals)  # 特征值由大到小排序，eigValInd十个arrary数组 1×n维
    eigValInd = eigValInd[:k]  # 选取前k个特征值的序号  1×k维
    redEigVects = eigVects[:,eigValInd] # 把符合条件的几列特征筛选出来组成P  n×k维
    lowDDataMat = meanRemoved * redEigVects  # 矩阵点乘筛选的特征向量矩阵  m×k维 公式Y=X*P
    PC_contribution = eigVals[eigValInd]/totalVar # 每个PC的贡献度，长度为k
    return lowDDataMat.real, PC_contribution.real


raw_data = scio.loadmat('data_EEG_AI.mat')['data']
raw_data = raw_data.transpose(2, 1, 0) # 7800x801x24
raw_data  =raw_data[:, 50:300, :]

num_PC = 25

data = np.empty([7800, num_PC, 24])

for i in range(0, 24):
    input_data = raw_data[:, :, i]
    data[:, :, i], PC_contribution = PCA(dataMat=input_data, k=num_PC)
    print('channel:', i, ' total contribution', sum(PC_contribution))

scio.savemat('rawdataPCA.mat', mdict={'data' : data, 'contribution' : PC_contribution})
