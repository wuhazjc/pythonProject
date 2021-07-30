import matplotlib.pyplot as plt
import numpy as np

# 永久用一个风格
plt.style.use("Solarize_Light2")

#==============折线图===================
#
# # =======[1]要不要plt.show()==================
# # .ipython中可用魔术方法 %matplotlib inline
# # pycharm 中必须使用 plt.show()
# x = [1, 2, 3, 4]
# y = [1, 4, 9, 16]
# plt.plot(x, y)
# plt.ylabel("squares")
# plt.xlabel("xlabel")
# # plt.show()
# # ============2.设置样式===============
# # 取出前5个
# print(plt.style.available[:5])
# # 临时改变风格，用到
# with plt.style.context("dark_background"):
#     plt.plot(x, y)
# # plt.show()
#
# # ===========3.将图像保存为文件==========
# x = np.linspace(0, 10, 100)
# plt.plot(x, np.exp(x))
# # plt.savefig("my_figure.png")
# #plt.show()
#
# # ========== Matlab 库 ============
#
# s = np.linspace(0, 2*np.pi, 100)  # 取0-2pi 之间等差数列的100个数  换成10个数，就可以看到折线了
# plt.plot(s, np.cos(s))
# plt.plot(s, np.sin(s))
# #plt.show()
#
# # 自己认为的：上面用了变量之后 下面的x不起作用，但是好像这样不对
# # plt记录的是上一个
# # 发现问题：上个plot了 但是没有show 猜想是show了之后才会释放
#
# # 【1】调整线条颜色和风格
# #   调整线条颜色
# offsets = np.linspace(0, np.pi, 5)
# colors = ["blue", "g", "r", "yellow", "pink"]
# for offset, color in zip(offsets, colors):
#     plt.plot(x, np.sin(x-offset), color=color)
# #plt.show()
#
# #    调整线条风格
# x = np.linspace(0, 10, 11)
# offsets = list(range(8))
# linestyles = ["solid", "dashed", "dashdot", "dotted", "-", "--", "-.", ":"]
# for offset, linestyle in zip(offsets, linestyles):
#     plt.plot(x, x+offset, linestyle=linestyle)
# # plt.show()
#
# # 调整线宽 ??print(linewidths)  ?? zip
# x = np.linspace(0, 10, 11)
# print(x)
# offsets = list(range(0, 12, 3))
# print(offsets)
# linewidths = (i*2 for i in range(1, 5))
# print(linewidths)
# for offset, linewidth in zip(offsets, linewidths):
#     plt.plot(x, x+offset, linewidth=linewidth)
# # plt.show()
#
#
# # 调整数据点标记
# x = np.linspace(0, 10, 11)
# offsets = list(range(0, 12, 3))
# markers = ["*", "+", "o", "s"]
# for offset, marker in zip(offsets, markers):
#     plt.plot(x, x+offset, marker=marker)
# # plt.show()
#
# # 调整数据点标记
# x = np.linspace(0, 10, 11)
# offsets = list(range(0, 12, 3))
# markers = ["*", "+", "o", "s"]
# for offset, marker in zip(offsets, markers):
#     plt.plot(x, x+offset, marker=marker, markersize=10)
# # plt.show()
#
# # 颜色跟风格设置的简写
# x = np.linspace(0, 10, 11)
# offsets = list(range(0, 8, 2))
# color_linestyles = ["g-", "b--", "k-.", "r:"]
# for offset, color_linestyle in zip(offsets, color_linestyles):
#     plt.plot(x, x+offset, color_linestyle)
# # plt.show()
#
# x = np.linspace(0, 10, 11)
# offsets = list(range(0, 8, 2))
# color_marker_linestyles = ["g*-", "b+--", "ko-.", "rs:"]
# for offset, color_marker_linestyle in zip(offsets, color_marker_linestyles):
#     plt.plot(x, x+offset, color_marker_linestyle)
# # plt.show()
#
# # 【2】调整坐标轴
# #  xlim 最大最小值
# x = np.linspace(0, 2*np.pi, 100)
# plt.plot(x, np.sin(x))
# plt.xlim(-1, 7)
# plt.ylim(-1.5, 1.5)
# # plt.show()
#
# # axis 可以把x轴和 y轴一起设定了
# x = np.linspace(0, 2*np.pi, 100)
# plt.plot(x, np.sin(x))
# plt.axis([-2, 8, -2, 2])
# plt.axis("tight") # 紧一些
# plt.axis("equal") # 扁平一些
#
# # plt.show()
#
# #?plt.axis  可以在ipython 中可以问一下
# # 对数坐标
# x = np.logspace(0, 5, 100)
# plt.plot(x, np.log(x))
# plt.xscale("log")  # 使用对数坐标 就是一条直线
# plt.show()

# # 调整坐标轴刻度
# x = np.linspace(0, 10, 100)
# plt.plot(x, x**2)
# plt.xticks(np.arange(0, 12, step=1), fontsize=15)
# plt.yticks(np.arange(0, 110, step=10))
# plt.tick_params(axis="both", labelsize=15) # 调整刻度样式
# plt.show()

# # 【3】设置图形标签
# x = np.linspace(0, 2*np.pi, 100)
# plt.plot(x, np.sin(x))
# plt.title("A Sine Curve", fontsize=20)
# plt.xlabel("x", fontsize=15)
# plt.ylabel("sin(x)", fontsize=15)
# plt.show()

# #【4】设置图例
# # 默认图例
# x = np.linspace(0, 2*np.pi, 100)
# plt.plot(x, np.sin(x), "b-", label="Sin")
# plt.plot(x, np.cos(x), "r--", label="Cos")
# plt.legend()
# plt.show()

# # 修饰图例
# x = np.linspace(0, 2*np.pi, 100)
# plt.plot(x, np.sin(x), "b-", label="Sin")
# plt.plot(x, np.cos(x), "r--", label="Cos")
# plt.ylim(-1.5, 2)
# plt.legend(loc="upper center", frameon=True, fontsize=15)
# plt.show()

# # 【5】添加文字和箭头
# #  添加文字
# x = np.linspace(0, 2*np.pi, 100)
# plt.plot(x, np.sin(x), "b-", label="Sin")
# plt.text(3.5, 0.5, "y=sin(x)", fontsize=15) # 文字坐标 文字 文字大小
# plt.show()

# #  添加箭头
# x = np.linspace(0, 2*np.pi, 100)
# plt.plot(x, np.sin(x), "b-", label="Sin")
# plt.annotate('local min', xy=(1.5*np.pi, -1), xytext=(4.5, 0)
#              ,arrowprops=dict(facecolor='black',shrink=0.1),
#              ) # 文字坐标 文字 文字大小
# plt.show()













