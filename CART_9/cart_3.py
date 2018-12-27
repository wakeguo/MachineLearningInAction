# 采用测试集对数据进行后剪枝

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


# 选择最好的特征以及特征下的分隔值
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 容许的误差下降值；切分的最小样本数
    tolS = ops[0]; tolN=ops[1]

    # 当前所有值相等，直接创建叶节点
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    # 默认最后一个特征为最佳切分特征， 计算其误差, 应该理解为所给数据的误差
    S = errType(dataSet)
    # 最好的误差值；最好的特征值；最好的分隔值
    bestS = float('inf'); bestIndex = 0; bestValue = 0
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
    # 分割后的样本数目太少，直接创建叶节点
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 创建回归树
def createTree(dataSet ,leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 用于判断输入变量是否是一棵树，也就是是否是叶节点
def isTree(obj):
    return (type(obj).__name__ == 'dict')
    # return isinstance(obj, dict)


# 对树进行塌陷处理，即返回树的平均值，计算两个叶节点的平均值，从下往上计算，最后返回一个平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0


# 对回归树进行裁剪合并
def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)  # 若测试数据是空的，则返回回归树的平均值

    if isTree(tree['right']) or isTree(tree['left']):  # 如果右子树或左子树非空，对测试数据划分
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

    if isTree(tree['left']):  # 若左树有子树，再进行递归遍历
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):  # 若右树有子树，再进行递归遍历
        tree['right'] = prune(tree['right'], rSet)

    if (not isTree(tree['left'])) and (not isTree(tree['right'])):  # 左右不含子树
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

        # 计算测试没有合并的误差，就是测试的每个数据减去原来数据的均值，再平方结果就是误差
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       np.sum(np.power(rSet[:, -1] - tree['right'], 2))

        # 训练数据没有分割的均值
        treeMean = (tree['left'] + tree['right'])/2.0

        # testData是没有分割的测试数据，减去训练数据的均值，再平方计算误差
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))

        # 然后就是对分割和不分割的误差进行比较
        if errorMerge < errorNoMerge:         # 1：testData是每次都在变化的。2：如何进行合并输出的
            print('Merging')
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__ == '__main__':
    dataMat = loadDataSet('ex2.txt')
    plotdata(dataMat)
    myMat = np.mat(dataMat)
    testdata = loadDataSet('ex2test.txt')
    testdata = np.mat(testdata)
    myTree = createTree(myMat, ops=(100, 4))
    print(myTree)
    newTree = prune(myTree, testdata)
    print(newTree)






