import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def main():
    # 训练数据
    X_train = np.array([[5, 4], [9, 6], [4, 7], [2, 3], [8, 1], [7, 2]])
    y_train = np.array([1, 1, 1, -1, -1, -1])
    X_new = np.array([[5, 3]])
    for k in range(1, 6, 2):  # 1,3,5
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1) # 构建实例
        # clf = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)  # distance 约近权重最大  n_jobs -1表示所有进程
        clf.fit(X_train,y_train) # 选择合适算法
        y_predict = clf.predict(X_new)
        print(clf.predict_proba(X_new))
        print("预测正确率：{:.0%}".format(clf.score([[5,3]],[[1]])))

        print("sha",clf.kneighbors(X_new, return_distance=False))
        print("sha", clf.kneighbors(X_new))
        print("graph",clf.kneighbors_graph(X_new, mode='distance')) #不懂这是啥
        #clf.kneighbors_graph(
        print("k={},被分类为：{}".format(k, y_predict))

if __name__ == "__main__":
    main()