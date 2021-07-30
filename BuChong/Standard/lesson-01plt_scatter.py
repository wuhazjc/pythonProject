import matplotlib.pyplot as plt
import numpy as np

# ==============13.1.2 散点图==============
# # 【1】简单散点图
# x = np.linspace(0, 2*np.pi, 20)
# plt.scatter(x, np.sin(x), marker="o", s=30, c="r") # marker:图的标记设置为圆圈 s大小 c颜色
# plt.show()

# # 【2】颜色配置 即变化的颜色
# x = np.linspace(0, 10, 100)
# y = x**2
# plt.scatter(x, y, c=y, cmap="inferno") # cmap="Blues" c=y 就是让c按照y进行映射 到cmap这组映射里面 Blues代表变化的颜色
# plt.colorbar() # 颜色条
# plt.show()

# 【3】 根据数据控制点的大小
# x, y, colors, size =(np.random.rand(100) for i in range(4))
#
# plt.scatter(x, y, c=colors, s=1000*size, cmap="viridis")
# #plt.colorbar()
# plt.show()

#【4】透明度
x, y, colors, size = (np.random.rand(100) for i in range(4))
plt.scatter(x, y, c=colors, s=1000*size, cmap="viridis", alpha=0.3)
plt.colorbar()
plt.show()

# 随机漫步设置画布的大小  有个随机漫步的例子需要看
plt.figure(figsize=(12, 6))