import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['lable'] = iris.target
    df.columns = ['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度', 'lable']
    data = np.array(df.iloc[:100, :])
    return data[:, :-1], data[:, -1]


class NaiveBayesClassifier:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.priori_P = {}
        self.conditional_P = {}
        self.N = {}
        pass

    def fit(self, X, y):
        for i in range(self.n_classes):
            # 公式 7.19
            self.priori_P[i] = (len(y[y == i]) + 1) / (len(y) + self.n_classes)
        for col in range(X.shape[1]):
            self.N[col] = len(np.unique(X[:, col]))  # 该特征值的个数
            self.conditional_P[col] = {}  # 存该列的特征取值为var,类别为i的条件概率
            for row in range(X.shape[0]):  # 取出该类特征不同的取值计算概率
                val = X[row, col]
                if val not in self.conditional_P[col].keys():  # 以列表返回一个字典所有的键。如果这个var没有计算过则计算
                    self.conditional_P[col][val] = {}
                    for i in range(self.n_classes):
                        D_xi = np.where(X[:, col] == val)  #
                        D_c = np.where(y == i)  # 返回符合条件的位置信息，即所在的行坐标
                        D_cxi = len(np.intersect1d(D_xi, D_c))  # 同样为行坐标
                        # 公式 7.20
                        self.conditional_P[col][val][i] = (D_cxi + 1) / (len(y[y == i]) + self.N[col])
                else:
                    continue

    def predict(self, X):
        pred_y = []
        for i in range(len(X)):
            p = np.ones((self.n_classes,))
            for j in range(self.n_classes):
                p[j] = self.priori_P[j]
            for col in range(X.shape[1]):
                val = X[i, col]
                for j in range(self.n_classes):
                    p[j] *= self.conditional_P[col][val][j]
            pred_y.append(np.argmax(p))
        return np.array(pred_y)


# 连续值
class NaiveBayesClassifierContinuous:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.priori_P = {}

    def fit(self, X, y):
        self.mus = np.zeros((self.n_classes, X.shape[1]))
        self.sigmas = np.zeros((self.n_classes, X.shape[1]))

        for c in range(self.n_classes):
            # 公式 7.19
            self.priori_P[c] = (len(y[y == c])) / (len(y))
            X_c = X[np.where(y == c)]

            self.mus[c] = np.mean(X_c, axis=0)
            self.sigmas[c] = np.std(X_c, axis=0)

    def predict(self, X):
        pred_y = []
        for i in range(len(X)):
            p = np.ones((self.n_classes,))
            for c in range(self.n_classes):
                p[c] = self.priori_P[c]
                for col in range(X.shape[1]):
                    x = X[i, col]
                    p[c] *= 1. / (np.sqrt(2 * np.pi) * self.sigmas[c, col]) * np.exp(
                        -(x - self.mus[c, col]) ** 2 / (2 * self.sigmas[c, col] ** 2))
            pred_y.append(np.argmax(p))
        return np.array(pred_y)


if __name__ == '__main__':
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(X_test)

    naive_bayes = NaiveBayesClassifierContinuous(n_classes=2)
    naive_bayes.fit(X_train, y_train)
    print('self.PrirP:', naive_bayes.priori_P)
    pred_y = naive_bayes.predict(X_test)
    print('pred_y:', pred_y)
    print('y_text:', y_test)

    x = np.array([i for i in range(len(pred_y))])
    plt.subplot(2, 1, 1)
    plt.scatter(x, pred_y, c='r')
    plt.subplot(2, 1, 2)
    plt.scatter(x, y_test, c='b')

    plt.show()
