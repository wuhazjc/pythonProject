import numpy as np
import pandas as pd


class NaiveBayes():
    def __init__(self, lambda_):
        self.lambda_ = lambda_  # 贝叶斯系数 取0时，即为极大似然估计
        self.y_types_count = None  # y的（类型：数量）
        self.y_types_proba = None  # y的（类型：概率）
        self.x_types_proba = dict()  # (xi 的编号，xi的取值，y的类型）：概率

    def fit(self, X_train, y_train):
        self.y_types = np.unique(y_train)  # y的所有取值类型
        print("y_types\n", self.y_types)
        X = pd.DataFrame(X_train)  # 转化成pandas DataFrame 数据格式，下同
        print("X DataFrame\n", X)
        y = pd.DataFrame(y_train)
        print("y DataFrame\n", y)
        self.y_types_count = y[0].value_counts()  # y的数量统计

        self.y_types_proba = (self.y_types_count + self.lambda_) / (y.shape[0] + len(self.y_types) * self.lambda_)

        print("X.columns\n", X.columns)
        for idx in X.columns:  # (xi的编号，xi的取值，y的类型）：概率的计算
            for j in self.y_types:  # 选取每一个y的类型
                p_x_y = X[(y == j).values][idx].value_counts()  # 选择所有y==j为真的数据点的第idx个特征的值，
                print("p_x_y\n", p_x_y)
                for i in p_x_y.index:  # 计算（xi的编号，xi的取值，y的类型）：概率的计算
                    self.x_types_proba[(idx, i, j)] = (p_x_y[i] + self.lambda_) / (
                            self.y_types_count[j] + p_x_y.shape[0] * self.lambda_)
        print(self.x_types_proba)

    def predict(self, X_new):
        res = []
        for y in self.y_types:  # 遍历y的可能取值
            p_y = self.y_types_proba[y]  # 计算y的先验概率 P(Y=ck)
            p_xy = 1
            for idx, x in enumerate(X_new):
                p_xy *= self.x_types_proba[(idx, x, y)]
            res.append(p_y * p_xy)
        for i in range(len(self.y_types)):
            print("[{}]对应概率：{:.2%}".format(self.y_types[i], res[i]))
        print(np.argmax(res))
        return self.y_types[np.argmax(res)]  # 取最大值的索引


def main():
    X_train = np.array(
        [[1, "S"], [1, "M"], [1, "M"], [1, "S"], [1, "S"], [2, "S"], [2, "M"], [2, "M"], [2, "L"], [2, "L"], [2, "L"],
         [3, "M"], [3, "M"], [3, "L"], [3, "L"]])
    y_train = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    print(X_train.shape)
    print(y_train.shape)
    clf = NaiveBayes(lambda_=0.2)
    clf.fit(X_train, y_train)
    X_new = np.array([2, "S"])
    y_predict = clf.predict(X_new)
    print("{}被分类为：{}".format(X_new, y_predict))


if __name__ == "__main__":
    main()
