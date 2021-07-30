import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)

# 求解步骤  1.确定模型  Model:  y = wx + b
#  2.选择损失函数  MSE
#  3.求解梯度并更新w，b   w = w - LR * w.grad b = b -LR * w.grad

lr = 0.1 # 学习率

# 创建训练数据
x = torch.rand(20, 1) * 10 # *data (tensor), shape=(20,1)  创建20个数据点
y = 2 * x + (5 + torch.randn(20, 1)) # y data (tensor),shape=(20,1)

# 构建线性回归参数
w = torch.randn((1), requires_grad=True) # 由于我们会用到自动梯度求导  所以会设置为True
b = torch.zeros((1), requires_grad=True)

for iteration in range(1000):
    #前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)  # 这里可以得到预测值

    # 计算 MSE loss
    loss =(0.5 * (y - y_pred) ** 2).mean()  # 真实值 - 预测值 的平方  0。5是求导过程中为了消掉平方而设置的

    # 反向传播
    loss.backward()  # 这样反向传播就可以得到梯度

    #更新参数
    b.data.sub_(lr * b.grad)  # 有了梯度就可以得到参数的更新  梯度*学习率 减掉这一项
    w.data.sub_(lr * w.grad)

    # 绘图
    if iteration % 20 == 0:
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, "Loss=%.4f" % loss.data.numpy(), fontdict={'size': 20, 'color': 'red' })
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration:{}\n w: {} b: {}".format(iteration, w.data.numpy(),b.data.numpy()))
        plt.pause(0.5)

        if loss.data.numpy() < 1:
            break
















