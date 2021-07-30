import numpy as np

# x = np.array([1, 2, 3, 4, 5])  # 用列表创建
# print(x)
# print(type(x))
# print(x.shape)  # 一维数组 5个元素
#
# x = np.array([[1, 2, 3],
#              [4, 5, 6],
#              [7, 8, 9]])
# print(x)
# print(x.shape) # 二维数组 3*3

# print(np.zeros(5, dtype=int))
# print(np.ones((2, 4), dtype=float))
# print(np.full((3, 5), 8.8))

# print(type(np.eye(3)))
print(np.arange(1, 15, 2)) # 创建一个线性序列数组 1，开始 15结束 步长为2

print(np.linspace(0, 1, 4)) # linspace line space 线性的排一组数 等差数列
print(np.logspace(0, 9, 10)) # 等比数列 10^0 10^9

# 8. 创建一个3*3，在0-1之间均匀分的的随机数构成的数组
print(np.random.random((3, 3)))
