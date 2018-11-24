# 机器学习实战训练营kNN快捷算法
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('iris.data.csv', header=None)
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']


# 获取数值是numpy
X = data.iloc[0:150, 0:4].values
y = data.iloc[0:150, 4].values

# 对类别进行赋值
y[y == 'Iris-setosa'] = 0
y[y == 'Iris-versicolor'] = 1
y[y == 'Iris-virginica'] = 2


X_setosa = X[0:50]; y_setosa = y[0:50]
X_versicolor = X[50:100]; y_versicolor = y[50:100]
X_virginica = X[100:150]; y_virginica = y[100:150]


plt.scatter(X_setosa[:, 0], X_setosa[:, 2], color='red', marker='o', label='setosa')
plt.scatter(X_versicolor[:, 0], X_versicolor[:, 2], color='blue', marker='^', label='versicolor')
plt.scatter(X_virginica[:, 0], X_virginica[:, 2], color='green', marker='s', label='virginica')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc=0)  # 注意其标签的显示
plt.show()


# training set
X_setosa_train = X_setosa[:40, :]
y_setosa_train = y_setosa[:40]
X_versicolor_train = X_versicolor[:40, :]
y_versicolor_train = y_versicolor[:40]
X_virginica_train = X_virginica[:40, :]
y_virginica_train = y_virginica[:40]
X_train = np.vstack([X_setosa_train, X_versicolor_train, X_virginica_train])
y_train = np.hstack([y_setosa_train, y_versicolor_train, y_virginica_train])

# # validation set   因为测试当中是没有使用到这组数据的，在选择最佳k值得时候就已经使用了交叉验证
# X_setosa_val = X_setosa[30:40, :]
# y_setosa_val = y_setosa[30:40]
# X_versicolor_val = X_versicolor[30:40, :]
# y_versicolor_val = y_versicolor[30:40]
# X_virginica_val = X_virginica[30:40, :]
# y_virginica_val = y_virginica[30:40]
# X_val = np.vstack([X_setosa_val, X_versicolor_val, X_virginica_val])
# y_val = np.hstack([y_setosa_val, y_versicolor_val, y_virginica_val])

# test set
X_setosa_test = X_setosa[40:50, :]
y_setosa_test = y_setosa[40:50]
X_versicolor_test = X_versicolor[40:50, :]
y_versicolor_test = y_versicolor[40:50]
X_virginica_test = X_virginica[40:50, :]
y_virginica_test = y_virginica[40:50]
X_test = np.vstack([X_setosa_test, X_versicolor_test, X_virginica_test])
y_test = np.hstack([y_setosa_test, y_versicolor_test, y_virginica_test])


# print('X_train:', X_train.shape)
# print('y_train:', y_train.shape)
# print('X_val:', X_val.shape)
# print('y_val:', y_val.shape)
# print('X_test:', X_test.shape)
# print('y_test:', y_test.shape)
# # X_train: (90, 4)
# # y_train: (90,)
# # X_val: (30, 4)
# # y_val: (30,)
# # X_test: (30, 4)
# # y_test: (30,)

print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)
# X_train: (120, 4)
# y_train: (120,)
# X_test: (30, 4)
# y_test: (30,)


class KNearestNeighbor(object):
    def __init__(self):
        pass

    # 训练函数
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    # 预测函数
    def predict(self, X, k=1):  # X表示X_test
        # 计算L2距离
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))  # 初始化距离函数
        # because(X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train, so
        d1 = -2 * np.dot(X, self.X_train.T)  # shape (num_test, num_train)
        d2 = np.sum(np.square(X), axis=1, keepdims=True)  # shape (num_test, 1)
        d3 = np.sum(np.square(self.X_train), axis=1)  # shape (1, num_train)
        dist = np.sqrt(d1 + d2 + d3)  # shape (num_test, num_train)
        # 根据K值，选择最可能属于的类别
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            dist_k_min = np.argsort(dist[i])[:k]  # 最近邻k个实例位置
            y_kclose = self.y_train[dist_k_min]  # 最近邻k个实例对应的标签
            y_pred[i] = np.argmax(np.bincount(y_kclose.tolist()))  # 找出k个标签中从属类别最多的作为预测类别
        return y_pred


# 交叉验证选择最好的k值
kNN = KNearestNeighbor()
K_accuracy = []
best_value = 0

num_folds = 6
K_classes = [k for k in range(2, 15)]
X_train_folds = np.split(X_train, num_folds)  # 默认是axis=0
y_train_folds = np.split(y_train, num_folds)
# y_train_folds=[array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=object), array

for k in K_classes:
    accuracies = []
    for i in range(num_folds):
        Xtr = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
        ytr = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
        Xcv = X_train_folds[i]
        ycv = y_train_folds[i]
        kNN.train(Xtr, ytr)
        ycv_pred = kNN.predict(Xcv, k=k)
        accuracy = np.mean(ycv_pred == ycv)
        accuracies.append(accuracy)
    accuracies_avg = np.mean(accuracies)
    K_accuracy.append(accuracies_avg)
    if accuracies_avg > best_value:
        best_value = accuracies_avg
        k_best = k

# 对比打印出最好的k值
for i in range(len(K_classes)):
    print('while k is: {}, accuray is: {:.5f}'.format(K_classes[i], K_accuracy[i]))
print('The best k is: {}'.format(k_best))


# Plot the cross validation
plt.plot(K_classes, K_accuracy, 'ro-')
plt.title('Cross-validation on k')
plt.xlabel('K')
plt.ylabel('Cross-validation accuracy')
plt.show()

# Test accuracy
kNN.train(X_train, y_train)
y_pred = kNN.predict(X_test, k=4)
accuracy = np.mean(y_pred == y_test)
print('\nThe test accuracy is: {}'.format(accuracy))









