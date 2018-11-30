import numpy as np
import matplotlib.pyplot as plt
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


# i是第一个alpha的下标，m是alpha的数目
def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


# 调整大于H小于L的alpha的值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


# 简化版SMO算法
# 数据集，类别标签，常数C，容错率，退出当前最大的循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0; m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b  # 一个数
            Ei = fXi - float(labelMat[i])  # 计算误差   # 下面的条件是不满足KKT条件
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()  # i_old
                alphaJold = alphas[j].copy()  # j_old
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:  # 已经在边界上了，不能再减小或者增大，因此不需要优化了。理解是支持向量了
                    print('L == H')
                    continue
                eta = 2 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print('j not moving enough')
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2
                alphaPairsChanged += 1
                print('iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas


# 得到w的值
def get_w(dataMat, labelMat, alphas):
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat).reshape(-1, 1)
    alphas = np.array(alphas)
    w = np.dot(dataMat.T, alphas * labelMat)
    return w.ravel()


# 进行绘图
def showClassifer(dataMat,labelMat, w, b):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus = np.array(data_plus)
    data_minus = np.array(data_minus)
    plt.scatter(data_plus[:, 0], data_plus[:, 1], s=30, color='blue', marker='o', alpha=0.7, label='Positive')
    plt.scatter(data_minus[:, 0], data_minus[:, 1], s=30, color='purple', marker='x', alpha=0.7, label='Negative')
    plt.legend(loc=0)
    plt.title('SVM')
    L = np.min(np.array(dataMat)[:, 0])
    H = np.max(np.array(dataMat)[:, 0])
    X = np.arange(L, H, 0.01)
    w1, w2 = w
    b = np.array(b).ravel()
    y = -(w1 * X + b)/w2
    plt.plot(X, y)
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter(x, y, s=100, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    w = get_w(dataArr, labelArr, alphas)
    showClassifer(dataArr, labelArr, w, b)










































