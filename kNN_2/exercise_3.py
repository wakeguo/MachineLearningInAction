from numpy import *

fr = open('datingTestSet2.txt')
arrayOlines = fr.readlines()
random.shuffle(arrayOlines)  # 打乱数据的排序
print(arrayOlines)
print(arrayOlines[:3])
numbersOfLines = len(arrayOlines)
returnMat = zeros((numbersOfLines, 3))
classLabelVector = []
index = 0
for line in arrayOlines:
    line = line.strip()
    print(line)
    listFromLine = line.split('\t')  # 两个数据之间是一个Tab
    print(listFromLine)
    returnMat[index, :] = listFromLine[0:3]
    classLabelVector.append(int(listFromLine[-1]))
    index += 1

print(returnMat[:10, :])
print(classLabelVector[:10])


minVals = returnMat.min(0)  # axis=0,纵的
maxVals = returnMat.max(0)
range = maxVals - minVals
normDataSet = zeros(shape(returnMat))  # shape实质为np.shape也可以为dataSet.shape
m = returnMat.shape[0]
normDataSet = returnMat - tile(minVals, (m, 1))  # 必须先创建一个为0的空的，然后赋值进去
normDataSet = normDataSet / tile(range, (m, 1))  # 数组是对应计算的

print(normDataSet[:10, :])
print(maxVals)
print(minVals)
print(range)
