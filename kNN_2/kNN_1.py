# 这个是机器学习实战课本的实例
from numpy import *
import operator
from os import listdir


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):  # 一个测试点数据，需要与全部的训练数据对比计算的
    """对数据预处理"""
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  # 行的
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):  # 要深刻理解k的含义，k就是取出的数据点的个数，取不同的量结果可能不一样
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 得到第一个的label


def file2matrix(filename):
    """获取数据"""
    fr = open(filename)
    arrayOlines = fr.readlines()
    random.shuffle(arrayOlines)  # 打乱数据的排序
    numbersOfLines = len(arrayOlines)
    returnMat = zeros((numbersOfLines, 3))  # 创建一个空的为0的然后赋值（列表也可以）进去才行
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')  # 两个数据之间是一个Tab
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector  # 一个是数组，一个是列表


def autoNorm(dataSet):  # 0-1之间
    """归一化数据"""
    minVals = dataSet.min(0)  # axis=0,纵的
    maxVals = dataSet.max(0)
    range = maxVals - minVals
    normDataSet = zeros(shape(dataSet))  # shape实质为np.shape也可以为dataSet.shape
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))  # 必须先创建一个为0的空的，然后赋值进去
    normDataSet = normDataSet/tile(range, (m, 1))  # 数组是对应计算的
    return normDataSet, range, minVals  # 后面两个数组都是1X3的


def datingClassTest():
    """分析错误率"""
    hoRatio = 0.1
    dataingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(dataingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 5)
        print("the classifier came back with: %d, the real answer is: %d"
              %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("the total error rete is: %f"%(errorCount/numTestVecs))


def classfiyPerson():
    """输入数据，得到结果"""
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?: '))
    ffMiles = float(input('frequent flier miles earned per year?: '))
    iceCream = float(input('liters of ice cream consumed per year?: '))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classfierResult = classify0((inArr-minVals)/ranges, norMat, datingLabels, 3)
    print('You will probably like this person:', resultList[classfierResult - 1])


def img2vector(filename):
    """把一个数字的数据整合到一行中"""
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classfierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d'
              % (classfierResult, classNumStr))
        if classfierResult != classNumStr:
            errorCount += 1
    print("\nthe total number if errors is {}".format(errorCount))
    print("\nthe total error rate is {:.2f}".format(errorCount/mTest))
















