"""
数据的基本处理方法
"""


import numpy as np
import matplotlib.pyplot as plt


# 读取数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


dataMat, labelMat = loadDataSet('ex0.txt')  # 数据都是列表

# np.array
dataMat = np.array(dataMat)  # shape(200, 2)
labelMat = np.array(labelMat)  # shape(200,)
X = dataMat[:, 1]  # shape(200,)
Y = labelMat  # shape(200,)
X_1 = X.reshape(-1, 1)  # shape(200, 1)
Y_1 = Y.reshape(-1, 1)  # shape(200, 1)
plt.scatter(X, Y, c='blue', s=30, alpha=0.5)  # 两种方法都对
plt.scatter(X_1, Y_1, c='blue', s=30, alpha=0.5)
plt.show()


# np.mat  对于scatter是不行的，但是列排对于plot是可以的。
dataMat = np.mat(dataMat)  # shape(200, 2)
labelMat = np.mat(labelMat)  # shape(1, 200)
X = dataMat[:, 1]  # shape(200, 1)
Y = labelMat.T  # shape(200, 1)

x = dataMat[:, 1].T  # shape(1,200)
y = labelMat  # shape(1,200)
plt.scatter(X, Y)  # 这两种方法都是错误的，需要转成array
plt.scatter(x, y)
plt.show()


dataMat = np.mat(dataMat)  # shape(200, 2)
labelMat = np.mat(labelMat)  # shape(1, 200)
index = dataMat[:, 1].argsort(axis=0)
dataMat = dataMat[index][:, 0, :]
labelMat = labelMat.T[index][:, 0, :]


X = dataMat[:, 1]  # shape(200, 1)
Y = labelMat  # shape(200, 1)
x = dataMat[:, 1].T  # shape(1,200)
y = labelMat.T  # shape(1,200)
# plt.scatter(X, Y)  # 这两种方法都是错误的，需要转成array
# plt.scatter(x, y)

plt.plot(X, Y)  # 可以
# plt.plot(x, y)  # 出错，没有图形显示


plt.show()


