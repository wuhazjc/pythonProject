import torch

torch.manual_seed(1)  # b不清楚这个是干什么的，估计待会会讲吧

# ================== example 1 =====================
# torch.cat
# flag = True
flag = False

if flag:
    t = torch.ones((2,3))

    t_0 = torch.cat([t, t], dim=0) # 在第0维上拼接 2+2=4，3
    t_1 = torch.cat([t, t], dim=1) # 在第1维上拼接 2，3+3=6

    print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))

# ================== example 2 ？？？ =====================

# torch.stack
# flag = True
flag = False

if flag:
    t = torch.ones((2, 5))

    t_stack = torch.stack([t, t], dim=2) # 第2维没有数据，在第2维上创建一个 进行拼接 2，5，2
    print("\n t_stack:{} shape:{}".format(t_stack, t_stack.shape))
    t_stack = torch.stack([t, t], dim=0) # 在第0维上新建一个  2，2，5
    print("\n t_stack:{} shape:{}".format(t_stack, t_stack.shape))

    t_stack = torch.stack([t, t, t], dim=0)  # 在第0维上新建一个  3，2，5  这个是为什么？？？
    print("\n t_stack:{} shape:{}".format(t_stack, t_stack.shape))
# ================== example 3 =====================
# torch.chunk 按照指定的维度进行平均切分
# flag = True
flag = False

if flag:
     a = torch.ones((2, 7))
     list_of_tensors = torch.chunk(a, dim=1, chunks=3) #第1个维度 5  5/2 向上取整为3

     for idx, t in enumerate(list_of_tensors):
         print("第{}个张量：{},shape is {}".format(idx+1, t, t.shape))
# ================== example 4 =====================
# torch.split 切分
# flag = True
flag = False

if flag:
    a = torch.ones((2, 5))
    #list_of_tensors = torch.split(a, 2, dim=1)  # 第1维切分的长度 [2,2] [2,2] [2,1]
    list_of_tensors = torch.split(a, [2, 1, 2], dim=1)  # 采用list 2+1+2=5 求和一定要等于指定维度上的长度
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{},shape is {}".format(idx + 1, t, t.shape))

# ================== 二、张量索引 example 5 =====================
# 2.1 torch.index_select() 功能：在维度dim上，按index索引数据 返回值：依index索引数据拼接的张量
# flag = True
flag = False

if flag:
    t = torch.randint(0, 9, size=(3, 3))
    idx =torch.tensor([0, 2],dtype=torch.long)  # 张量中常用的类型是long 和float
    t_select = torch.index_select(t, dim=0, index=idx)
    print("t:\n{}t_select:\n{}".format(t, t_select))

# ================== 二、张量索引 example 6 =====================
# 2.2 torch.masked_select() 功能：按mask中的true进行索引 返回值：一维张量
flag = True
# flag = False

if flag:
    t = torch.randint(0, 9, size=(3, 3))
    mask = t.ge(5) #ge is mean greater than or equal / gt:greater than le lt
    t_select = torch.masked_select(t, mask)
    print("t:\n{}\n mask:\n{}\n t_select:\n{}",t, mask, t_select)

# ================== 三、张量变换 example 7 =====================
# 3。1 torch.reshape() 功能：变换张量形状 注意事项：当张量在内存中是连续时，新张量与input共享数据内存
# flag = True
flag = False

if flag:
    t = torch.randperm(8)  # 创建一个一维的张量 功能是随机打乱一个数字序列。其内的参数决定了随机数的范围。
    t_reshape = torch.reshape(t, (2, 4))
    print("t:{}\nt_reshape:\n{}".format(t, t_reshape))

    t[0] = 1024
    print("t:{}\nt_reshape:\n{}".format(t, t_reshape))
    print("t.data 内存地址：{}".format(id(t.data)))
    print("t_reshape.data 内存地址：{}".format(id(t_reshape.data)))

# ================== 三、张量变换 example 8 =====================
# 3。2 torch.transpose() 功能：交换张量的两个维度
# 3。3 torch.t()   功能：2维张量转置，对矩阵而言，等价于torch.transpose(input, 0, 1)

# flag = True
flag = False

if flag:
    t = torch.rand((2, 3, 4))
    t_transpose = torch.transpose(t,dim0=1,dim1=2)
    print("t shape:{}\nt_transpose shape:{}".format(t.shape, t_transpose.shape))

    t_rand = torch.rand((2, 3))  #只针对于两维的  为了代码更简洁
    t_t = torch.t(t_rand)
    print("t_rand shape:{}\nt_transpose shape:{}".format(t_rand.shape, t_t.shape))

# ================== 三、张量变换 example 9 =====================
# 3。4 torch.squeeze() 功能：压缩长度为1的维度 dim 为none移除长度为1的  dim=1 移除当且仅当长度为1的
# 3。5 torch.unsqueeze() 功能：依据dim扩展维度  指定维度就是1

# flag = True
flag = False

if flag:
    t = torch.rand((1, 2, 3, 1))
    t_sq = torch.squeeze(t)        # 移除所有维度是1的
    t_0 = torch.squeeze(t, dim=0)  # 第0维度是1所以移除
    t_1 = torch.squeeze(t, dim=1)  # 如果第一维是1则移除 如果不是则不移除  因为不是所以不移除
    print(t.shape)
    print(t_sq.shape)
    print(t_0.shape)
    print(t_1.shape)

# ================== 张量数学运算 example 10 =====================
# 一。加减乘除
# 二。对数，指数，幂函数
# 三。三角函数
# torch.add()  功能：逐元素计算 input（第一个张量）+ alpha（乘项因子） * other（第二个张量）
# torch.addcdiv() torch.addcmul() 在优化过程中会经常使用到
flag = True
# flag = False

if flag:
    t_0 = torch.randn((3, 3))  # 标准正态
    t_1 = torch.ones_like(t_0)
    t_add = torch.add(t_0, 10, t_1)
    print("t_0:\n{}\nt_1:{}\nt_add_10:\n{}".format(t_0, t_1, t_add))








