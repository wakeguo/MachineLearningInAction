# 手写识别系统（机器学习实战训练营）

import numpy as np
import operator
from os import listdir


# 定义将一个图像转换为向量函数
def img2vector(filename):
    returnVect = np.zeros((1, 1024))  # 存储图片像素的向维度是1X1024
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])  # 图片尺寸是32X32,将其以此放入向量returnVect
    return returnVect


# 定义k近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    daffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = daffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 定义手写数字识别系统函数
def handwritingClassTest():
    hwLabel = []
    trainingFileList = listdir('trainingDigits')  # 导入训练集
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]  # fileNameStr得到的是每个文件的名称，例如‘0_0.txt'
        fileStr = fileNameStr.split('.')[0]  # 去掉'.txt',剩下'0_0'
        classNumStr = int(fileStr.split('_')[0])  # 按下划线'_'划分'0_0'，取第一个元素为类别标签
        hwLabel.append(classNumStr)  # 数据集标签
        trainingMat[i, :] = img2vector('trainingDigits/{}'.format(fileNameStr))  # 数据集
    # 测试样本
    testFileList= listdir('testDigits')  # 读取全部的测试集
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]  # 测试集的一个
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/{}'.format(fileNameStr))  # 测试集的一个换算成了向量
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabel, 3)  # 调用kNN函数
        print('The classifier came back with: {}, the real answer is: {}'.format(classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1
    print('\nThe total number of error is: {}'.format(errorCount))
    print('\nThe total error rate is: {}'.format(errorCount/float(mTest)))


handwritingClassTest()










































