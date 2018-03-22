import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
# 获取适应度值
def ObjValue(gAddress):
    objvalue = zeros((n,1))
    for i in range(n):
        ## 函数1：
        objvalue[i,0] = 100 * power(power(gAddress[i,0], 2) - gAddress[i,1], 2) + power(1 - gAddress[i,0], 2)
    return objvalue
def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total
def cumsum(fit_value):
    for i in range(len(fit_value) - 2, -1, -1):
        # print(i)
        t = 0
        j = 0
        while (j <= i):
            t += fit_value[j]
            j += 1
        fit_value[i] = t
        fit_value[len(fit_value) - 1] = 1
    # print(fit_value)
def best(fit_value):
    px = len(fit_value)
    index_best = 0
    best_fit = fit_value[0]
    for i in range(1, px):
        if(fit_value[i] > best_fit):
            best_fit = fit_value[i]
            index_best =i
    return [best_fit,index_best]
# 获取两点范数
def norm (x,y):
    return sqrt(power(x[0] - y[0], 2) + power(x[1] - y[1], 2))
def range1(X,Y):
    k = len(Y)
    t = []
    add = []
    for i in range(k):
        t.append(X[:,i])
    t = np.array(t)
    # print(X,"s")
    # print(t)
    # print(len(t[0]))
    # print(k)
    # print(t[0][1])
    for i in range(k):
        for j in range(len(t[0])):
            if(t[i][j]<Y[i][0]):
                t[i][j] = Y[i][0]
            if(t[i][j]>Y[i][1]):
                t[i][j] = Y[i][1]
    for i in range(k):
        X[:,i] = t[i,:]
if __name__ == '__main__':
    generation = 0  # 迭代次数
    # 初始化参数
    domx = [[-2.048,2.048],[-2.048,2.048]]      # 解空间
    rho = 0.4                   # 荧光素挥发因子
    gamma = 0.6                 # 适应度提取比例
    beta = 0.08                 # 领域变化率
    nt = 5                      # 邻居阈值(邻居萤火虫数)
    s = 0.03                    # 移动步长
    l0 = 5                      # 初始荧光素值
    rs = 2.048                      # 最大决策半径
    r0 = 2.048                      # 初始动态决策半径
    maxGeneration = 200         # 最大迭代次数
    #
    # # 分配空间
    dim = len(domx)             # 解空间维度
    n = 80                      # 种群规模
    gAddress = np.zeros((n,dim))   # 存放萤火虫地址空间
    gValue = np.zeros((n,1))       # 存放适应度值空间
    li = np.zeros((n,1))           # 存放荧光素值空间
    rdi = np.zeros((n,1))          # 存放决策半径空间
    value = []                  # 存放最优值，用于作图
    gen = []                    # 存放迭代次数，用于作图
    # # 初始化地址
    for i in range(n):
        for j in range(dim):
            gAddress[i,j] = domx[j][0] + (domx[j][1] - domx[j][0]) * random.random()
    ## 初始荧光素值
    li[:,0] = l0
    # print(li)
    # ## 初始决策半径
    rdi[:,0] = r0
    objvalue0 = ObjValue(gAddress)
    best_fit, index_best = best(objvalue0)
    currentbestfit = best_fit
    currentbestindex = index_best
    l0, = plt.plot(gAddress[:, 0], gAddress[:, 1], '+')  # 传hadle需要，
    plt.legend(handles=[l0], labels=['class1'], loc='best')
    plt.show()
    value.append(float(currentbestfit))
    # ## 迭代开始
    while (generation < maxGeneration):
        generation += 1

        ## 更新荧光素
        objvalue1 = ObjValue(gAddress)
        for i in range(n):
            li[i,0] = (1 - rho) * li[i,0] + gamma * objvalue1[i,0]
        ## 各萤火虫开始移动
        for i in range(n):
            Nit = []
            ## 存放萤火虫序号
            for j in range(n):
                # if (j != i):
                if((norm(gAddress[j,:],gAddress[i,:]) < rdi[i]) and (li[i,0] < li[j,0])):
                    Nit.append(j)
            # 寻找下一个移动点
            num = len(Nit)
            if num != 0:
                Nitioti = []
                numerator = []
                Pij = []
    #             ## 获取邻域内所有点的荧光素值
                for j in range(num) :
                    Nitioti.append(li[Nit[j]])
                Nitioti = np.array(Nitioti)
    #             ## 求荧光素和
                sumNitioti = sum(Nitioti)
                ## 获取萤火虫移动概率公式的分子和分母
                for j in range(num):
                    numerator.append(Nitioti[j] - li[i,0])
                denominator = sumNitioti - li[i,0]
                ## 移动概率
                for j in range(num):
                    Pij.append(numerator[j] / denominator)
                Pij = np.array(Pij)
                ## 概率归一化
                cumsum(Pij)
    #             ## 确定移动位置
                Pos = []

                for j in range(num):
                    if random.random() < Pij[j]:
                        pos = j
                        break
                j = Nit[pos]
                ## i点向j点移动
                gAddress[i,:] = gAddress[i,:] + s * ((gAddress[j,:] - gAddress[i,:]) / norm(gAddress[j,:],gAddress[i,:]))
                ## 限制i点的移动范围
                range1(gAddress, domx)

            ## 更新决策半径
            rdi[i] = rdi[i] + beta * (nt - len(Nit))
            rdi[i] = min(rs,max(0.1,rdi[i]))
        ## 获取最优个体
        objectvalue2 = ObjValue(gAddress)
        best_fit1, index_best1 = best(objectvalue2)
        if (best_fit1 > currentbestfit):
            currentbestfit = best_fit1
        if (best_fit1 > currentbestfit):
            currentbestfit = objectvalue2[index_best1]
        # print(generation , "+", currentbestfit)
        value.append(float(currentbestfit))
        # print(gAddress[index_best1,:])
    # print(objectvalue2)
    # print(value)
    # ll, = plt.plot(gAddress[:, 0], gAddress[:, 1], '+')  # 传hadle需要，
    # plt.legend(handles=[ll], labels=['class2'], loc='best')
    # plt.show()
        # value.append(float(curBestVal))
        # gen.append(generation)
        # print("迭代次数：{}，最优个体所在位置:{},最优值：{}".format(generation,gAddress[curBestIndex,:],curBestVal))
        # generation += 1
    # print(objectValue)
    # plt.figure(2)
    # plt.plot(gen,value)
    # plt.xlabel("Generation")
    # plt.ylabel("BestValue")
    # plt.show()
    #
    x = np.linspace(0,200,201)
    plt.plot(x,value)
    plt.show()


