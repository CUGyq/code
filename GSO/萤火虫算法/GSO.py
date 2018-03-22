#! /bin/user/env python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
# 获取适应度值
def ObjValue(gAddress):
    objValue = zeros((n,1))
    for i in range(n):
        ## 函数1：
        objValue[i,0] = 100 * power(power(gAddress[i,0], 2) - gAddress[i,1], 2) + power(1 - gAddress[i,0], 2)
        ## 函数2：
        # Q = exp(-(power(gAddress[i,0], 2) + power(gAddress[i,1] + 1, 2)))
        # W = exp(-(power(gAddress[i,0], 2) + power(gAddress[i,1], 2)))
        # E = exp(-(power(gAddress[i,1], 2) + power(gAddress[i,0] + 1, 2)))
        # objValue[i,0] = (3 * (1 - power(gAddress[i,0], 2)) * Q) - (10 * (gAddress[i,0] / 5 - power(gAddress[i,0], 3) - power(gAddress[i,1], 5)) * W) - (1 / 3 * E)
    return objValue

# 获取两点范数
def norm (x,y):
    return sqrt(power(x[0] - y[0], 2) + power(x[1] - y[1], 2))

if __name__ == '__main__':
    # 初始化参数
    domx = [[-2.048,2.048],[-2.048,2.048]]      # 解空间
    # domx = [[-2 * pi,2 * pi],[-2 * pi,2 * pi]]
    rho = 0.4                   # 荧光素挥发因子
    gamma = 0.6                 # 适应度提取比例
    beta = 0.08                 # 领域变化率
    nt = 5                      # 邻居阈值(邻居萤火虫数)
    s = 0.03                    # 移动步长
    l0 = 5                      # 初始荧光素值
    rs = 3                      # 最大决策半径
    r0 = 3                      # 初始动态决策半径
    maxGeneration = 200         # 最大迭代次数

    # 分配空间
    dim = len(domx)             # 解空间维度
    n = 50                      # 种群规模
    gAddress = zeros((n,dim))   # 存放萤火虫地址空间
    gValue = zeros((n,1))       # 存放适应度值空间
    li = zeros((n,1))           # 存放荧光素值空间
    rdi = zeros((n,1))          # 存放决策半径空间
    generation = 0              # 迭代次数
    value = []                  # 存放最优值，用于作图
    gen = []                    # 存放迭代次数，用于作图
    # 初始化地址
    for i in range(n):
        for j in range(dim):
            gAddress[i,j] = domx[j][0] + (domx[j][1] - domx[j][0]) * random.random()
    # print("萤火虫地址为：\n",gAddress)

    # # 多模态函数
    # ## 设置三维坐标
    fig = plt.figure(1)
    ax = Axes3D(fig)
    ## 生成数据
    n2 = 256
    ## 函数1：
    x = np.linspace(-2.048,2.048,n2)
    y = np.linspace(-2.048,2.048,n2)
    ## 函数2：
    # x = np.linspace(-3, 3)
    # y = np.linspace(-3, 3)
    X,Y = np.meshgrid(x,y)
    R = power(x,2)
    Z = 100 * power(R - Y,2) + power(1 - X,2)
    # Q= exp(-(power(X , 2) + power(Y + 1 , 2)))
    # W= exp(-(power(X , 2) + power(Y , 2)))
    # E= exp(-(power(Y,2) + power(X + 1 , 2)))
    # Z = (3 * (1 - power(X , 2)) * Q) - (10 * (X / 5 - power(X , 3) - power(Y , 5) * W)) - (1 / 3 * E)
    ## 画3D图
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
    plt.show()

    # 参数初始化
    ## 初始荧光素值
    li[:,0] = 10
    ## 初始决策半径
    rdi[:,0] = r0
    ## 迭代开始
    generation = 1
    while (generation <= maxGeneration):
        ## 更新荧光素
        objValue = ObjValue(gAddress)
        # print("目标函数值:",objValue)
        li[:,0] = (1 - rho) * li[:,0] + gamma * objValue[:,0]
        # print("update li:",li)
        ## 各萤火虫开始移动

        for i in range(n):
            Nit = []
            ## 存放萤火虫序号
            for j in range(n):
                if (j != i):
                    if((norm(gAddress[j,:],gAddress[i,:]) < rdi[i]) and (li[i,0] < li[j,0])):
                        Nit.append(j)
            # print("Nit:",Nit)
            ## 寻找下一个移动点
            num = len(Nit)
            if num != 0:
                Nitioti = []
                numerator = []
                Pij = []
                ## 获取邻域内所有点的荧光素值
                for x in range(num) :
                    Nitioti.append(li[Nit[x]])
                Nitioti = array(Nitioti)
                # print("Nitioti:",(Nitioti))
                ## 求荧光素和
                sumNitioti = sum(Nitioti)
                ## 获取萤火虫移动概率公式的分子和分母
                for x in range(num):
                    numerator.append(Nitioti[x] - li[i,0])
                # print("numerator:",numerator)
                denominator = sumNitioti - li[i,0]
                # print(denominator)
                ## 移动概率
                for x in range(num):
                    Pij.append(numerator[x] / denominator)
                ## 概率归一化
                for x in range(1,num):
                    Pij[x] = Pij[x - 1] + Pij[x]
                # print(Pij,"s")
                for x in range(num):
                    Pij[x] = Pij[x] / Pij[num - 1]
                # print(Pij,"s")
                ## 确定移动位置
                for x in range(num):
                    Ps = random.random()
                    # print(Ps)
                    if Ps < Pij[x]:
                        pos = x
                        break
                j = Nit[pos]
                ## i点向j点移动
                gAddress[i,:] = gAddress[i,:] + s * ((gAddress[j,:] - gAddress[i,:]) / norm(gAddress[j,:],gAddress[i,:]))
                ## 限制i点的移动范围
                dim = len(gAddress[0])
                for x in range(dim):
                    if (gAddress[i,x] >= 2.048):
                        gAddress[i,x] = 2.048
                    if (gAddress[i,x] <= -2.048):
                        gAddress[i,x] = -2.048
            ## 更新决策半径
            rdi[i] = rdi[i] + beta * (nt - len(Nit))
            rdi[i] = min(rs,max(0,rdi[i]))
        ## 获取最优个体
        objectValue = ObjValue(gAddress)
        bestValue = objectValue[0]
        for i in range(1,n):
            if bestValue < objectValue[i]:
                bestValue = objectValue[i]
                bestIndex = i

        if (generation == 1):
            curBestVal = bestValue
            curBestIndex = bestIndex
        else:
            if curBestVal < bestValue:
                curBestVal = bestValue
                curBestIndex = bestIndex
        value.append(float(curBestVal))
        gen.append(generation)
        print("迭代次数：{}，最优个体所在位置:{},最优值：{}".format(generation,gAddress[curBestIndex,:],curBestVal))
        generation += 1
    print(objectValue)
    plt.figure(2)
    plt.plot(gen,value)
    plt.xlabel("Generation")
    plt.ylabel("BestValue")
    plt.show()



