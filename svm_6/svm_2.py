import numpy as np
import random


# 载入数据和标签
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


dataMat, labelMat = loadDataSet('testSet.txt')

labelMat = np.array(labelMat).reshape(-1, 1)
dataMat = np.array(dataMat)

# print(labelMat.shape)
# print(dataMat)
print(np.max(dataMat[:, 0]))













