"""
预测鲍鱼的年龄，没有加入偏置项
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


# 标准线性回归
def standRegres(xArr, yArr):
    xMat = np.mat(xArr)  # shape(200, 2)
    yMat = np.mat(yArr).T  # shape(200, 1)
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


# 局部加权线性回归
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
    return yHat


# 测试误差
def ressError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()  # 列表减去一维的array，结果也是一维array


# 对数据的基本分析
# abX, abY = loadDataSet('abalone.txt')
# yHat01 = lwlTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
# yHat1 = lwlTest(abX[0:99], abX[0:99], abY[0:99], 1)
# yHat10 = lwlTest(abX[0:99], abX[0:99], abY[0:99], 10)
#
# a = ressError(abY[0:99], yHat01)
# print(a)
# ws = standRegres(abX[0:99], abY[0:99])
# yHat = np.mat(abX[100:199]) * ws
# b = ressError(abY[100:199], yHat.T.A)
# print(b)


# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(xMat.shape[1])*lam
    if np.linalg.det(denom) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMean)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, xMat.shape[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat


abX, abY = loadDataSet('abalone.txt')
print(abX)
wMat = ridgeTest(abX, abY)
# print(wMat[0, :])  # lam很小时的岭回归，会发现与下面的标准回归w值基本一样
#
#
# xMat = np.mat(abX)
# yMat = np.mat(abY).T
# xMean = np.mean(xMat, 0)
# xVar = np.var(xMat, 0)
# xMat = (xMat - xMean)/xVar
# yMean = np.mean(yMat, 0)
# yMat = yMat - yMean
# w = standRegres(xMat, yMat.T)
# print(w)  # 归一化之后的标准回归分析
#
# plt.plot(wMat)
# plt.show()


# 前向逐步回归
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMean)/xVar
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    m, n = xMat.shape
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1)); wsMax = ws.copy()
    for i in range(numIt):
        # print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssE = ressError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


abX, abY = loadDataSet('abalone.txt')
returnMat = stageWise(abX, abY, 0.001, 5000)
print(returnMat[-1, :])  # 前向逐步回归，会发现与下面的标准回归w值基本一样


xMat = np.mat(abX)
yMat = np.mat(abY).T
xMean = np.mean(xMat, 0)
xVar = np.var(xMat, 0)
xMat = (xMat - xMean)/xVar
yMean = np.mean(yMat, 0)
yMat = yMat - yMean
w = standRegres(xMat, yMat.T)
print(w)  # 归一化之后的标准回归分析

plt.plot(returnMat)
plt.show()













