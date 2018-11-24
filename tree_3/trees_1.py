from math import log
import operator


# 创建数据和标签
def creatDataSet():
    """获取数据和标签"""
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']  # 代表着按特征分类
    return dataSet, labels


# 经验熵的计算
def calcShannonEnt(dataSet):
    """获取数据集标签的经验熵"""  # 就是sum(p * log(p))
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        # labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1  # 和上面三行功能一样
    # labelCounts = {'yes': 2, 'no': 3}  # 结果
    shannonEnt = 0.0
    for key in labelCounts:  # .keys()可以省略的
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt  # 数据集的经验熵，就是原始的数据值，是按标签分类的。


myData, myLabel = creatDataSet()
print('myData: ', myData)
print('myLabel:', myLabel)
print('经验熵：', calcShannonEnt(myData))
# myData:  [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
# myLabel: ['no surfacing', 'flippers']
# 经验熵： 0.9709505944546686


# 按照指定的特征进行分类
def splitDataSet(dataSet, axis, value):  # axis代表着每一个特征的位置也就是索引的位置值，按这个特征分类
    """获取按指定的特征进行分类"""           # value代表着对应特征的分类值，就是一共有多少类，不同的类就是不同的value
    retDataSet = []                      # dataSet代表待划分的数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]  # 获取featVec的axis之前的值
            reduceFeatVec.extend(featVec[axis+1:])  # 获取featVec的axis之后的值,然后组成一个列表
            retDataSet.append(reduceFeatVec)  # 然后去掉按分类的值之后再合并在一个嵌套的列表中
    return retDataSet


print(splitDataSet(myData, 0, 1))
print(splitDataSet(myData, 1, 1))
# [[1, 'yes'], [1, 'yes'], [0, 'no']]
# [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']]


# 选择最好的数据集划分方式，返回最大的信息增益的特征分类的位置，只分了一次
def chooseBestFeatureToSplit(dataSet):
    """计算信息增益，返回最大的信息增益的特征分类的位置"""
    numFeatures = len(dataSet[0]) - 1  # 得出一共有多少特征，减去1是表示减去标签，此处的数据集是2
    baseEntropy = calcShannonEnt(dataSet)  # 获取数据集标签的经验熵
    bestInfoGain = 0.0  # 因为0是分界线，小于0就是按数据集标签分类了
    bestFeature = -1  # 代表数据集的标签索引值
    for i in range(numFeatures):  # i代表索引，就是按哪一个特征进行分类
        featList = [example[i] for example in dataSet]  # featList就是一个列表，只是每次的循环列表的值不一样
                                                        # 获取第i个特征里面所有的值，就是分类值，纵向的
        uniqueVals = set(featList)  # 变成集合，去除第i个特征里面重复值，就是分了多少类
        newEntrop = 0.0
        # 计算信息增益
        for value in uniqueVals:  # value代表着在一个特征里面按那个特征值分类，因为有不用的分类值
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntrop += prob * calcShannonEnt(subDataSet)  # 计算的是按一个特征分类的经验条件熵
        infoGain = baseEntropy - newEntrop  # 计算信息增益，就是数据集的经验熵减去一个特征分类的经验条件熵
        # print('第{}个信息增益{:.3f}'.format(i, infoGain))
        if infoGain > bestInfoGain:  # 因为小于0，直接按数据集的标签进行分类了，索引位置为-1。
            bestInfoGain = infoGain
            bestFeature = i  # 返回最大的信息增益的特征分类的位置
    return bestFeature  # 返回最大的信息增益的特征分类的位置


print('最好的特征分类位置: ', chooseBestFeatureToSplit(myData))
# 最好的特征分类位置:  0


# 多数表决法确定该叶子节点的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 返回最多的一类


# 递归构建决策树
def createTree(dataSet, labels):
    """得出决策树的字典嵌套"""
    classList = [example[-1] for example in dataSet]  # 得到所有的数据的标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 递归第一个停止条件是所有的类标签全部相同，直接返回类标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 使用完了所有特征，任然不能将数据划分成仅包含唯一类别的分组，采用多数原理进行分类
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 得到最好的分类特征的位置
    bestFeatLabel = labels[bestFeat]  # 由位置得到最好的分类特征

    myTree = {bestFeatLabel: {}}  # 返回第一个分类特征
    del(labels[bestFeat])  # 然后删除第一个分类特征
    featValues = [example[bestFeat] for example in dataSet]  # 提取出第一个分类特征里面所有的属性值，纵的
    uniqueVals = set(featValues)  # 去除重复的属性值
    for value in uniqueVals:  # value表示按最好特征中每个的分类值
        subLabels = labels[:]  # 剩下的全部特征
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # 这里是一个函数的递归计算。myTree中bestFeatLabel, value实际都是keys, values是函数返回的值。value是循环值的
        # 在for循环中运行，构成一个递归运算。
    return myTree


# print('myTree: ', createTree(myData, myLabel))
# # myTree:  {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# print(myLabel)  # 有这个subLabels = labels[:]结果是['flippers']，没有这个结果是[]


# 对给的数据进行预测
def classify(inputTree, featLabels, testVec):  # 输入分别是决策树，特征列表，预测的数据
    # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    firstStr = list(inputTree.keys())[0]  # 第一个分类特征
    secondDict = inputTree[firstStr]  # 决策树的第二个字典，就是第一个分类特征对应的值
    featIndex = featLabels.index(firstStr)  # 第一个分类特征在特征列表中对应的索引位置

    # 第二种：改进的算法
    key = testVec[featIndex]  # 直接提取测试集中的值
    valueFeat = secondDict[key]
    if isinstance(valueFeat, dict):
        classLabel = classify(valueFeat, featLabels, testVec)
    else:
        classLabel = valueFeat
    return classLabel
    # 第一种：课本预测方法
    # for key in secondDict.keys():  # 先遍历再判断
    #     if testVec[featIndex] == key:
    #         if type(secondDict[key]).__name__ == 'dict':
    #             classLabel = classify(secondDict[key], featLabels, testVec)
    #         else:
    #             classLabel = secondDict[key]
    # return classLabel


mytree = createTree(myData, myLabel)
myData, myLabel = creatDataSet()
print(classify(mytree, myLabel, [1, 0]))
print(classify(mytree, myLabel, [1, 1]))


# 使用决策树预测隐形眼镜类型
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
print(lensesTree)


# 使用决策树进行预测
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
classeslabel = classify(lensesTree, lensesLabels, lenses[0][:-1])
print('实际值:', lenses[0][-1])
print('预测值:', classeslabel)


preds = []
for i in range(len(lenses)):
    pred = classify(lensesTree, lensesLabels, lenses[i][:-1])
    preds.append(pred)
print(preds)


















