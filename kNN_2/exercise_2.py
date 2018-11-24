import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


group, labels = kNN.createDataSet()
a = kNN.classify0([0, 0], group, labels, 3)


datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print(datingDataMat[:10, :])
print(datingLabels[:10])

x1_label = []
y1_label = []
x2_label = []
y2_label = []
x3_label = []
y3_label = []

for i in range(len(datingLabels)):
    if datingLabels[i] == 1:
        x1_label.append(datingDataMat[i][0])
        y1_label.append(datingDataMat[i][1])
    elif datingLabels[i] == 2:
        x2_label.append(datingDataMat[i][0])
        y2_label.append(datingDataMat[i][1])
    elif datingLabels[i] == 3:
        x3_label.append(datingDataMat[i][0])
        y3_label.append(datingDataMat[i][1])

matplotlib.rcParams['font.family'] = 'STSong'
matplotlib.rcParams['font.size'] = 10

plt.scatter(x1_label, y1_label, s=30, c='r', marker='+', label='不喜欢')  # s != size
plt.scatter(x2_label, y2_label, s=20, color='green', label='魅力一般')
plt.scatter(x3_label, y3_label, s=10, color='blue', label='极具魅力')

# plt.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
#           15.0*array(datingLabels), 15.0*array(datingLabels))  # 对应标签的颜色

plt.xlabel('每年获取的飞行常客里程数', fontproperties='SimHei', fontsize=15)
plt.ylabel('玩视频游戏所消耗的时间百分比', fontproperties='SimHei', fontsize=15)
plt.legend()
plt.show()


normMat, range, minVals = kNN.autoNorm(datingDataMat)


print(normMat.shape)
print(range)
print(minVals)
