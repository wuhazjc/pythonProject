import numpy as np
import matplotlib.pyplot as plt


# 构建MyPerceptron
class MyPerceptron:
    def __init__(self):
        self.w = None
        self.b = 0
        self.l_rate = 1

    def fit(self, X_train, y_train):
        # 用样本点的特征数更新初始w，如x1=（3，3），有两个特征，则self.w=[0,0]
        self.w = np.zeros(X_train.shape[1])
        i = 0
        while i < X_train.shape[0]:
            X = X_train[i]
            y = y_train[i]  # 如果y*(wx+b)<=0,说明是误判点，更新w，b
            if y * (np.dot(self.w, X) + self.b) <= 0:
                self.w = self.w + self.l_rate * np.dot(y, X)
                self.b = self.b + self.l_rate * y
                i = 0  # 如果是误判点，从头进行检测
            else:
                i += 1


def draw(X, w, b):
    X_new = np.array([[0], [6]])  # 生产分离超平面上的两点
    y_predict_false = -(b + w[0] * X_new) / w[1]
    print(y_predict_false)
    # y_predict =w.T @ (X_new) + b
    y_predict = -(b + w[0] * X_new) / w[1]
    print("ypredict", y_predict)
    plt.plot(X[:2, 0], X[:2, 1], "g*", label="1")
    plt.plot(X[2:, 0], X[2:, 1], "rx", label="-1")
    plt.plot(X_new, y_predict, "b-")
    plt.axis([0, 6, 0, 6])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def main():
    # 构造训练数据集 numpy array
    X_train = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    print(X_train.shape)
    # 构建感知机对象，对数据集继续训练
    perceptron = MyPerceptron()
    perceptron.fit(X_train, y)

    # 结果图像绘制
    draw(X_train, perceptron.w, perceptron.b)

if __name__ == "__main__":
    main()