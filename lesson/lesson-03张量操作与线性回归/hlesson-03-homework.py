import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)

lr = 0.01  # 学习率    20191015修改
best_loss = float("inf") #表示正无穷


# 1. 调整线性回归模型停止条件以及y = 2*x + (5 + torch.randn(20, 1))中的斜率，训练一个线性回归模型；
#
# 2.计算图的两个主要概念是什么？
#
# 3.动态图与静态图的区别是什么？


# 创建训练数据
x = torch.rand(200, 1) * 10  # x data (tensor), shape=(20, 1)
print(x)

y = 2*x + (5 + torch.randn(20, 1))  # y data (tensor), shape=(20, 1)

# 构建线性回归参数
w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

for iteration in range(1000):
    # 前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算 MSE loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # 清零张量的梯度   20191015增加
    w.grad.zero_()
    b.grad.zero_()

    # 绘图
    if iteration % 20 == 0:
        plt.cla()   # 防止社区版可视化时模型重叠2020-12-15
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)

        if loss.data.numpy() < 1:
            break
    plt.show()