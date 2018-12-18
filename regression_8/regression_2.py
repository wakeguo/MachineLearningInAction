"""
局部加权线性回归
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


# 绘制数据
def plotdata(dataMat, labelMat):
    dataMat = np.array(dataMat)  # shape(200, 2)
    labelMat = np.array(labelMat)
    X = dataMat[:, 1]  # shape(200,)
    Y = labelMat  # shape(200,)
    plt.scatter(X, Y, c='blue', s=30, alpha=0.5)
    plt.show()


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = xMat.shape[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2 * k**2))

    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))

    return testPoint * ws


def lwlTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat  # shape(m, )


dataMat, labelMat = loadDataSet('ex0.txt')
xArr = np.array(dataMat)  # shape(200, 2)
yArr = np.array(labelMat)  # shape(200,)   但是np.mat(labelMat) = (1,200) 然后.T=(200,1)
# xArr[:, 1].shape = (200,)  # 这里一定得注意会变成一维的
plt.scatter(xArr[:, 1], yArr, c='blue', alpha=0.7, s=30)


y = lwlTest(dataMat, dataMat, labelMat, k=0.003)
xMat = np.mat(dataMat)  # shape(200, 2)
yMat = np.mat(labelMat)  # shape(1. 200)

srtInd = xMat[:, 1].argsort(0)
# xSort = xMat[srtInd]  # shape(200, 1, 2)
xSort = xMat[srtInd][:, 0, :]  # shape(200, 2)
# print(xSort[:, 1])  #shape(200,1)
# print(y.shape)  # shape(200,)
# print(y[srtInd].shape)  # shape(200, 1)  使用argsort是会增加一维的
plt.plot(xSort[:, 1], y[srtInd], c='red')
plt.show()






