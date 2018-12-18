"""
线性回归
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
    X = dataMat[:, 1]  # shape(200,)  但是对于mat会是shape(200,1)
    Y = labelMat  # shape(200,)
    plt.scatter(X, Y, c='blue', s=30, alpha=0.5)
    plt.show()


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)  # shape(200, 2)
    yMat = np.mat(yArr).T  # shape(200, 1)
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def plotws(dataMat, labelMat, ws):
    dataMat = np.array(dataMat)  # shape(200, 2)
    labelMat = np.array(labelMat)  # shape(200,)
    plt.scatter(dataMat[:, 1], labelMat, c='blue', s=30, alpha=0.5)  # dataMat[:,1].shape=(200,)
    ws = np.array(ws)
    dataMatcopy = dataMat.copy()
    dataMatcopy.sort(0)  # 需要对数据进行从小到大排序
    y = np.dot(dataMatcopy, ws)
    plt.plot(dataMatcopy[:, 1], y, c='red')
    yHat = np.dot(dataMat, ws)
    yMat = np.array(labelMat).reshape(-1, 1).T
    a = np.corrcoef(yHat.T, yMat)
    print(a)
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('ex0.txt')
    plotdata(dataMat, labelMat)
    ws = standRegres(dataMat, labelMat)
    plotws(dataMat, labelMat, ws)














