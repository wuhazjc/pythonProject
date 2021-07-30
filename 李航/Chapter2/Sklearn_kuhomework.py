from sklearn.linear_model import Perceptron
import numpy as np

# 训练数据集
X_train = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, -1])

# 构建Perceptron对象，训练数据并输出结果
# perceptron = Perceptron()
# # 最大迭代次数，终止条件
# perceptron = Perceptron(max_iter=1000, tol=1e-3)
# 学习率 eta0 默认为1
perceptron = Perceptron(eta0=0.5, max_iter=1000, tol=1e-3)

perceptron.fit(X_train, y)

print("w:", perceptron.coef_, "\n", "b:", perceptron.intercept_, "\n", "n_iter", perceptron.n_iter_)
# 测试模型预测的准确率
res = perceptron.score(X_train, y)
print("correct rate:{:.0%}".format(res))
