# 简单的数据分割，分成两类

import numpy as np
import matplotlib.pyplot as plt


# 载入数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


# 对数据按照特征和分割值进行分割
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 绘制数据图形
def plotdata(dataMat):
    dataMat = np.array(dataMat)
    X = dataMat[:, 0]
    y = dataMat[:, 1]
    plt.scatter(X, y, c='b', s=20, alpha=0.6)
    plt.title('DataSet')
    plt.show()


# 生成叶节点， 返回目标变量的均值
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


# 误差估计，返回的是总方差，所以均方差要乘以数据集中样本的个数
def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 容许的误差下降值；切分的最小样本数
    tolS = ops[0]; tolN=ops[1]
    # 当前所有值相等，则退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    # 默认最后一个特征为最佳切分特征， 计算其误差, 应该理解为所给数据的误差
    S = errType(dataSet)
    # 最好的误差值；最好的特征值；最好的分隔值
    bestS = float('inf'); bestIndex = 0; bestValue = 0  # np.inf
    # 不包含最后一列的，最后一列是输出值，不是特征
    for featIndex in range(n-1):  # 遍历所有特征
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):  # 遍历所有的特征下的分割值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)  # 计算分割后的误差
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    if (S - bestS) < tolS:  # 分割后的误差没有降低太多，就直接创建叶节点
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


if __name__ == '__main__':
    dataMat = loadDataSet('ex00.txt')
    plotdata(dataMat)
    myMat = np.mat(dataMat)
    bestIndex, bestValue = chooseBestSplit(myMat)
    print('bestIndex:', bestIndex)
    print('bestValue:', bestValue)












