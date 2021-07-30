import torch
import numpy as np

# 1.   安装anaconda,pycharm, CUDA+CuDNN（可选），虚拟环境，pytorch，并实现hello pytorch查看pytorch的版本
print(torch.__version__)


# 2.   张量与矩阵、向量、标量的关系是怎么样的？
    # 张量为一个多维数组，是标量、向量、矩阵的高维拓展
    # Tensor（张量）类似于Numpy的 ndarray，但还可以在GPU上使用来加速运算
        # 标量（Scalar）：0阶（r=0）张量
        # 向量（Vector）：1阶（r=1）张量
        # 矩阵（Matrix）：2阶（r=2）张量

# 3.   Variable“赋予”张量什么功能？
    # 0.4.0版本后Variable已并入Tensor
    # Variable 是torch.autograd中的数据类型，主要用于封装Tensor，进行自动求导

    # data:被包装的Tensor
    # grad：data的梯度
    # grad_fn:创建Tensor的Function，是自动求导的关键
    # requires_grad:指示是否需要梯度
    # is_leaf:指示是否叶子结点

# 4.   采用torch.from_numpy创建张量，并打印查看ndarray和张量数据的地址；
        # 注意事项：从torch.from_numpy创建的tensor与原ndarray共享内存，当修改其中一个数据时，另外一个也将改动

flag = False
if flag:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(arr, id(arr))     # 4501321744

    t = torch.from_numpy(arr)
    print(t, id(t.data))    # 4501321744
    # 使用clone方法拷贝张量, 拷贝后的张量和原始张量内存独立
    arr = t.clone().numpy() # 也可以使用tensor.data.numpy()

    arr[0,0]=99
    print(arr, id(arr))
    print(t, id(t.data))

# print(id(arr))
# 5.   实现torch.normal()创建张量的四种模式。

# 四种模式
# 1。mean为标量，std为标量
# 2。mean为标量，std为张量
# 3。mean为张量，std为标量
# 4。mean为张量，std为张量

# 1。mean为标量，std为标量
flag = False

if flag:
    mean = 0
    std = 1
    t_normal = torch.normal(mean, std, size=(4,))
    print(mean, std, t_normal)

# 2。mean为标量，std为张量      这里用了broadcast机制
if flag:
    mean = 1
    std = torch.arange(1, 5, dtype=torch.float)
    t_normal = torch.normal(mean, std)
    print(mean, std, t_normal)

# 3。mean为张量，std为标量
if flag:
    mean = torch.arange(1, 5, dtype=torch.float)
    std =1
    t_normal = torch.normal(mean,std)
    print(mean, std, t_normal)

# 4。mean为张量，std为张量
mean = torch.arange(1, 5, dtype=torch.float)
std = torch.arange(1, 5, dtype=torch.float)
t_normal = torch.normal(mean,std)
print(mean,std, t_normal)










