# -*- coding: utf-8 -*-
"""
# @file name  : lesson-05-Logsitic-Regression.py
# @author     : tingsongyu
# @date       : 2019-09-03 10:08:00
# @brief      : 逻辑回归模型训练
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)

# 对数几率回归就是逻辑回归
# 分类任务是交叉墒

# ============================ step 1/5 生成数据 ============================
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias      # 类别0 数据 shape=(100, 2)
y0 = torch.zeros(sample_nums)                         # 类别0 标签 shape=(100)
x1 = torch.normal(-mean_value * n_data, 1) + bias     # 类别1 数据 shape=(100, 2)
y1 = torch.ones(sample_nums)                          # 类别1 标签 shape=(100)
train_x = torch.cat((x0, x1), 0)  # 自变量拼接
train_y = torch.cat((y0, y1), 0)  # 因变量拼接


# ============================ step 2/5 选择模型 ============================
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x


lr_net = LR()   # 实例化逻辑回归模型


# ============================ step 3/5 选择损失函数 ============================
loss_fn = nn.BCELoss()  # 选择二分类的交叉墒函数

# ============================ step 4/5 选择优化器   ============================
lr = 0.01  # 学习率
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9) # 随机梯度下降

# ============================ step 5/5 模型训练 ============================
for iteration in range(1000):  # 训练迭代更新

    # 前向传播
    y_pred = lr_net(train_x)  # 前向传播 训练数据输入到模型得到输出

    # 计算 loss
    loss = loss_fn(y_pred.squeeze(), train_y) # 将模型的输出和标签同时输入给损失函数

    # 反向传播  自动的梯度求导
    loss.backward()

    # 更新参数  优化器更新权值
    optimizer.step()

    # 清空梯度
    optimizer.zero_grad()

    # 绘图
    if iteration % 20 == 0:

        # 绘图的时候我们会计算一个分类准确率（在讲解张量的时候讲过） float 改变类型 squeeze压缩张量维度
        mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类  ge用来生成一个mask>=0.5的时候true
        correct = (mask == train_y).sum()  # 计算正确预测的样本个数
        acc = correct.item() / train_y.size(0)  # 计算分类准确率

        # 绘制训练数据
        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        # 绘制逻辑回归模型
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()

        plt.show()
        plt.pause(0.5)

        # 模型停止条件
        if acc > 0.99:
            break