import numpy as np
import matplotlib.pyplot as plt


# 载入数据 自己定义的数据
def loadSimpData():
    dataMat = np.mat([[1, 2.1],
                      [1.5, 1.6],
                      [1.3, 1],
                      [1, 1],
                      [2, 1]])
    classLabels = [1, 1, -1, -1, 1]
    return dataMat, classLabels

# 载入数据 产生的都是列表
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))
    return dataMat, labelMat


# 绘制数据图形
def plotdataMat(dataMat, classLabels):
    dataMat_positive = []
    dataMat_negative = []
    m, n = np.shape(dataMat)
    for i in range(m):
        if classLabels[i] == 1:
            dataMat_positive.append(dataMat[i].A1)  # .A1很重要,转换成一维的array
        else:
            dataMat_negative.append(dataMat[i].A1)
    dataMat_positive = np.array(dataMat_positive)
    dataMat_negative = np.array(dataMat_negative)
    plt.scatter(dataMat_negative[:, 0], dataMat_negative[:, 1], c='red', marker='o', s=30, label='negative')
    plt.scatter(dataMat_positive[:, 0], dataMat_positive[:, 1], c='blue', marker='x', s=30, label='positive')
    plt.legend(loc=0)
    plt.show()


# 数据集，维度，阈值，两侧方向的选择(此函数只进行了一侧操作)
def stumpClassfiy(dataMatrix, dimen, threshVal, threshIneq):
    retArry = np.ones((np.shape(dataMatrix)[0], 1))  # 对所有数据标签设置为1
    if threshIneq == 'lt':  # less than
        retArry[dataMatrix[:, dimen] <= threshVal] = -1   # 进行了多项操作
    else:
        retArry[dataMatrix[:, dimen] > threshVal] = -1
    return retArry  # 分类好的标签


# 单层决策树生成函数
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10  # 分的份数
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):  # 对维度进行遍历
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max()  # 对应维度的最小值和最大值
        stepSize = (rangeMax - rangeMin)/numSteps  # 每一份的步长
        for j in range(-1, int(numSteps) + 1):  # 前后都多出一个份数，对阈值进行遍历
            for inequal in ['lt', 'gt']:  # 对阈值两边进行遍历
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassfiy(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # 采用权重和标签计算误差值
                # print('Split: dim %d, thresh %.2f, inequal %s, weights error %.3f'
                #       % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst  # 字典信息，最小误差，最好的分类标签


# D = np.mat(np.ones((5, 1))/5)
# dataMat, classLabels = loadSimpData()
# plotdataMat(dataMat, classLabels)
# bestStump, minError, bestClassEst = buildStump(dataMat, classLabels, D)
# print('bestStump: ', bestStump)
# print('minError: ', minError)
# print('bestClassEst: ', bestClassEst)


# 基于单层决策树的AdaBoost训练过程
# 数据集，标签，最大迭代次数
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = np.shape(dataArr)[0]  # dataArr是列表
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print('D: ', D.T)
        alpha = float(0.5 * np.log((1 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print('classEst: ', classEst)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha * classEst  # 这几行程序和统计学习方法上面的一样
        # print('aggClassEst: ', aggClassEst)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum()/m  # 每次的误差值
        print('total error: ', errorRate)
        if errorRate == 0:
            break
    return weakClassArr, aggClassEst  # 每次迭代的信息列表，最终的预测标签值


# dataMat, dataLabels = loadSimpData()
# weakClassArr = adaBoostTrainDS(dataMat, dataLabels, 9)
# print(weakClassArr)


# 进行预测的AdaBoost函数
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassfiy(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])  # 需要得出每次的基本分类器
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print('aggClassEst: ', aggClassEst)
    return np.sign(aggClassEst)


# dataMat, dataLabels = loadSimpData()
# classifierArr = adaBoostTrainDS(dataMat, dataLabels, 30)
# correctrate = adaClassify([[5, 5], [0, 0]], classifierArr)
# print(correctrate)


# dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
# classifierArray, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
# testdataArr, testlabelArr = loadDataSet('horseColicTest2.txt')
# predicition = adaClassify(testdataArr, classifierArray)
# m = len(testlabelArr)
# errArr = np.ones((m, 1))
# errArrnumer = errArr[predicition != np.mat(testlabelArr).T].sum()
# print(errArrnumer/m)


# 绘制ROC曲线
def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0
    numPosClas = np.sum(np.array(classLabels) == 1)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1:
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0
            ySum += cur[1]
        plt.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], 'b-')  # 注意图形坐标对应的绘制
        cur = (cur[0] - delX, cur[1] - delY)
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    plt.show()
    print('the Area Under the Curve is: ', ySum * xStep)


dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
classifierArray, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 50)
plotROC(aggClassEst.T, labelArr)








