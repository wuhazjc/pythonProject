# -*- coding: utf-8 -*-
"""
# @file name  : lesson-05-autograd.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-08-30 10:08:00
# @brief      : torch.autograd
"""
import torch
torch.manual_seed(10)


# ====================================== retain_graph ==============================================
flag = True
# flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward(retain_graph=True) # 如果不设置这个参数，如果调用两个backward就会出现问题，因为在第一个backward时候已经将计算图释放掉了。
    # print(w.grad)
    y.backward()

# ====================================== grad_tensors ==============================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)     # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)
    y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2

    loss = torch.cat([y0, y1], dim=0)       # [y0, y1]  拼接
    grad_tensors = torch.tensor([1., 2.])  #这个是设置权重  w.grad=y0*1+y1*2

    loss.backward(gradient=grad_tensors)    # gradient 传入 torch.autograd.backward()中的grad_tensors

    print(w.grad)


# ====================================== autograd.gard ==============================================
# flag = True
flag = False
if flag:

    x = torch.tensor([3.], requires_grad=True)  # 二阶导数的求导
    y = torch.pow(x, 2)     # y = x**2

    grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2 * 3 = 6  output=y input=x  用于创建导数的计算图，对导数再次求导
    print(grad_1)  # 是一个元组

    grad_2 = torch.autograd.grad(grad_1[0], x)              # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
    print(grad_2)


# ====================================== tips: 1  梯度不会自动清零 ==============================================
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        w.grad.zero_() # 下划线表示inplace操作  就是原位操作


# ======================== tips: 2。依赖于叶子结点的结点，requires_grad默认为True ==============================================
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)


# ====================================== tips: 3 ==============================================
# flag = True
flag = False
if flag:

    a = torch.ones((1, ))
    print(id(a), a)

    # a = a + torch.ones((1, ))
    # print(id(a), a)

    a += torch.ones((1, ))
    print(id(a), a)


# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    w.add_(1)  # 不可以做inplace操作 例子：柜子是地址，里面的东西换了，再去找对应的东西就不对了。 很奇怪的解释 我觉得有点问题。
    """
    autograd小贴士：
        梯度不自动清零 
        依赖于叶子结点的结点，requires_grad默认为True     
        叶子结点不可执行in-place 
    """
    y.backward()