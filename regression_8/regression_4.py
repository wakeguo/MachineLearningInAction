import numpy as np
from bs4 import BeautifulSoup
import random


# 采用爬虫知识获取数据
def scrapePage(retX, retY, setHtml, yr, numPce, origPrc):
    """
    :param retX: 数据x
    :param retY: 数据y
    :param setHtml: HTML文件
    :param yr: 年份
    :param numPce:乐高部件数目
    :param origPrc: 原价
    :return:
    """
    with open(setHtml, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)

    while (len(currentRow) != 0):
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0

        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)

            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)


def setDataCollect(retX, retY):
    scrapePage(retX, retY, 'setHtml/lego8288.html', 2006, 800, 49.99)  # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, 'setHtml/lego10030.html', 2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, 'setHtml/lego10179.html', 2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, 'setHtml/lego10181.html', 2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, 'setHtml/lego10189.html', 2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, 'setHtml/lego10196.html', 2009, 3263, 249.99)


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


# 测试误差
def ressError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()  # 列表减去一维的array，结果也是一维array



def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal, 30))
    for i in range(numVal):
        trainX = []; trainY = []
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < 0.9 * m:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = np.mat(testX); matTrainX = np.mat(trainX)
            meanTrainX = np.mean(matTrainX, 0)
            varTrainX = np.var(matTrainX)
            matTestX = (matTestX - meanTrainX)/varTrainX
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)  # np.mean()对列表也能操作
            errorMat[i, k] = ressError(yEst.T.A, testY)
    meanErrors = np.mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0); varX = np.var(xMat, 0)
    nuReg = bestWeights/varX
    print('the best model from Ridge Regression is: \n', nuReg)




lgX = []
lgY = []
setDataCollect(lgX, lgY)
lgX1 = np.ones((np.shape(lgX)[0], 5))  # 也有插入的方法看看
lgX1[:, 1:5] = np.mat(lgX)
w = standRegres(lgX1, lgY)
print('{} + {}*年份 + {}*数目 + {}*是否全新 + {}*原价'.format(w[0], w[1], w[2], [3], w[4]))  # 找一下解决
print('%f %+f*年份 %+f*部件数量 %+f*是否为全新 %+f*原价' % (w[0], w[1], w[2], w[3], w[4]))
print(w)

print(np.shape(lgX))
print(np.shape(lgY))

crossValidation(lgX, lgY, 10)


