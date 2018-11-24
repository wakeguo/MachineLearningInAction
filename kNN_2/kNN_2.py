# 使用k近邻算法改进网站的配对效果（机器学习实战训练营）

import numpy as np
import operator


# 定义数据集导入函数
def file2matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}  # 三个类别
    fr = open(filename)  # 打开文件
    arrayOLines = fr.readlines()  # 读取全部的行，逐行打开
    numberOfLines = len(arrayOLines)  # 文件的行数
    returnMat = np.zeros((numberOfLines, 3))  # 初始化特征矩阵
    classLabelVector = []  # 初始化输出标签向量
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 40920	8.326976	0.953952	3
        # 如果没有这个步骤下部的结果是['40920', '8.326976', '0.953952', '3\n']
        listFormLine = line.split('\t')  # ['40920', '8.326976', '0.953952', '3']
        returnMat[index, :] = listFormLine[0:3]  # 得到数据集
        if (listFormLine[-1].isdigit()):  # 判断listFormLine最后一个元素是数字，是True
            classLabelVector.append(int(listFormLine[-1]))
        else:  # listFormLine最后一个元素是字符串
            classLabelVector.append(love_dictionary.get(listFormLine[-1]))
        index += 1
    return returnMat, classLabelVector  # 返回数据数组，类别标签是1，2，3


# 对特征进行归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))  # shape是元组的
    # m = dataSet.shape[0]
    # normDataSet = dataSet - np.tile(minVals, (m, 1))
    # normDataSet = normDataSet/np.tile(ranges, (m, 1))
    normDataSet = dataSet - minVals  # 广播运算更简单一点
    normDataSet = normDataSet/ranges
    return normDataSet, ranges, minVals  # 特征归一化的数据，数据范围，数据最小值


# 定义k近邻算法
def classify0(inX, dataSet, labels, k):  # inX是测试集，dataSet是训练集，labels是训练样本标签，k是取的最近邻个数
    dataSetSize = dataSet.shape[0]  # 训练样本个数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 重复n次
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5  # distance是inX与dataSet的欧式距离
    sortedDistIndices = distances.argsort()  # 返回数值从小到大的索引值排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 返回k近邻中所属类别最多的一类


# 测试算法：作为完整程序验证分类器
def datingClassTest():
    hoRatio = 0.1  # 数据的10%用来测试
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 导入数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 特征归一化
    m = normMat.shape[0]  # 样本个数
    numTestVecs = int(m*hoRatio)  # 测试样本个数
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('The classifier came back with: {}, the real answer is: {}'.format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print('The total error is: {:.3f}'.format(errorCount/float(numTestVecs)))  # 错误率
    print('Error counts is {}'.format(errorCount))  # 错误个数


datingClassTest()


# 根据用户输入，在线判断类别
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print('You will probably like this person: {}'.format(resultList[classifierResult - 1]))


classifyPerson()



















