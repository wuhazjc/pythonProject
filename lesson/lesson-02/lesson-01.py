import torch
import numpy as np

# ===========1-01 第一周张量简介与创建ß
# =========== 一，直接创建 ==================
# =========== example 1 ==================
#flag = True
flag = False
if flag:
    arr = np.ones((3, 3))
    print("ndarray的数据类型：", arr.dtype)

    t = torch.tensor(arr, device='cpu')
    #t = torch.tensor(arr)
    print(t)
# =========== example 2 ==================
# 通过 torch。from_numpy 创建张量
# flag = True
flag = False

if flag:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    t = torch.from_numpy(arr)
    print("numpy array:", arr)
    print("tensor :", t)

    arr[0][0] = 0
    print("numpy array:", arr)
    # print("tensor :", t)
    # 为了印证 from_numpy 是np和torch共享内存
# =========== 二、依据数值创建 ==================
# =========== example 3 ==================
# 通过torch.zeros 创建张量 全0 全1 ones
# flag = True
flag = False
if flag:
    out_t = torch.tensor([1])
    print(out_t)

    t = torch.zeros((3, 3), out=out_t)

    print(out_t)
    print(id(t), id(out_t), id(t) == id(out_t))
    # 同一个数据 命名不同 那为什么要用这个
# =========== example 4 ==================
# 通过torch.zeros_like 创建张量
# 通过torch.full 创建张量
# flag = True
flag = False
if flag:
    t = torch.full((3, 3), 10)
    print(t)
# =========== example 5 ==================
# torch.arrange() 创建等差的1维向量  区间[start,end)  数列公差 默认为1
# flag = True
flag = False
if flag:
    t = torch.arange(2, 10, 2)
    print(t)

# =========== example 6 ==================
# torch.linspace() 创建均分数列  区间[start,end] steps 数列长度  end-start/steps
# flag = True
flag = False
if flag:
    t = torch.linspace(2, 10, 3)
    print(t)
# torch.logspace() 创建对数均分数列  区间[start,end] steps 数列长度  底为base
# flag = True
flag = False
if flag:
    t = torch.logspace(2, 10, 3, 10.0)
    print(t)
# =========== example 7 ==================
# torch.eye() 创建单位对角矩阵(2维张量）  默认维方阵
#  n: 矩阵行数（通常之设置n）  m 矩阵列数
flag = True
# flag = False
if flag:
    t = torch.eye(2)
    print(t)
# =========== 三、依概率分布创建 ==================
# 3.1 torch.normal()  生成正态分布（高斯分布）
# mean :均值  std：标准差  四种模式  标量和张量
# =========== example 8 ==================
# flag = True
flag = False

if flag:
    # mean:张量 std：张量
    # mean = torch.arange(1, 5, dtype=torch.float)
    # std = torch.arange(1, 5, dtype=torch.float)
    # t_normal = torch.normal(mean, std)
    # print("mean:{}\nstd:{}".format(mean, std))
    # print(t_normal) # 均值为1 标准差为1的正态分布采样而来


    # mean：标量 std ：标量
    # t_normal = torch.normal(0., 1., size=(4,))
    # print(t_normal) # 4个元素都是从均值为0，标准差为1 的正态分布中采样得来

    # mean：张量 std ：标量
    mean = torch.arange(1, 5, dtype=torch.float)
    std = 1
    t_normal =torch.normal(mean, std)
    print("mean:{}\nstd:{}".format(mean, std))
    print(t_normal) # 均值长度为4 标准差为1的正态分布采样而来

# 3.2 torch.randn() 3.3 torch.randn_like()   生成标准正态分布（高斯分布）
# 3.4 torch.rand() 3.5 torch.rand_like()   功能：在区间[0,1) 上，生成均匀分布
# 3.6 torch.randint() 3.7 torch.randint_like()   功能：在区间[low,high) 上，生成整数均匀分布
# 3.8 torch.randperm() 功能：生成从0到n-1的随机排列 用来生成乱序索引 3.9 torch.bernoulli()   以input为概率生成伯努利分布（0-1分布，两点分布）
# =========== example 9 ==================
