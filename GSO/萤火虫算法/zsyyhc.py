import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
# 获取适应度值
def ObjValue(gAddress):
    objValue = zeros((n,1))
    # print(gAddress)
    for i in range(n):
        # 函数1：
        objValue[i,0] = 100 * power(power(gAddress[i,0], 2) - gAddress[i,1], 2) + power(1 - gAddress[i,0], 2)
    return objValue

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
    s0 = 0.1                    # 移动步长
    smax = 0.16
    l0 = 5                      # 初始荧光素值
    rs = 2.048                      # 最大决策半径
    r0 = 2.048                      # 初始动态决策半径
    maxGeneration = 100         # 最大迭代次数
    #
    # # 分配空间
    dim = len(domx)             # 解空间维度
    n = 80                      # 种群规模
    gAddress = np.zeros((n,dim))   # 存放萤火虫地址空间

    gValue = np.zeros((n,1))       # 存放适应度值空间
    li = np.zeros((n,1))           # 存放荧光素值空间
    rdi = np.zeros((n,1))          # 存放决策半径空间
    s = np.zeros((n,1))            #移动步长
    # value = []                  # 存放最优值，用于作图
    # gen = []                    # 存放迭代次数，用于作图
    # # 初始化地址
    for i in range(n):
        for j in range(dim):
            gAddress[i,j] = domx[j][0] + (domx[j][1] - domx[j][0]) * random.random()
    ## 初始荧光素值
    li[:,0] = l0
    # print(li)
    # ## 初始决策半径
    rdi[:,0] = r0
    #初始化移动步长
    s[:,0] = s0
    ADD = np.zeros((n,dim))
    gAddress1 = np.zeros((n,dim))
    for i in range(n):
        # gAddress0[i,:] = gAddress[i,:]
        gAddress1[i,:] = gAddress[i,:]
    objvalue0 = ObjValue(gAddress)
    best_fit, index_best = best(objvalue0)
    currentbestfit = best_fit
    currentbestindex = index_best
    # print(currentbestfit)
    ## 迭代开始
    value = []  # 存放最优值，用于作图
    value.append(float(currentbestfit))
    value.append(float(currentbestfit))
    generation = 1
    while (generation < maxGeneration):
        generation+=1
        objvalue1 = ObjValue(gAddress1)
        for i in range(n):
            li[i, 0] = (1 - rho) * li[i, 0] + gamma * objvalue1[i, 0]
                    #     ## 各萤火虫开始移动
        for i in range(n):
            Nit = []
            ## 存放萤火虫序号
            for j in range(n):
                if ((norm(gAddress1[j, :], gAddress1[i, :]) < rdi[i]) and (li[i, 0] < li[j, 0])):
                     Nit.append(j)
                    # 寻找下一个移动点
            num = len(Nit)
            # print(num)
            if num == 0:
                # ADD[i, :] = gAddress1[i, :]
                ADD[i,:] = gAddress1[i,:]
                s[i][0] = s[i][0]
                # print(i)
            else:
                Nitioti = []
                numerator = []
                Pij = []
        ## 获取邻域内所有点的荧光素值
                for j in range(num):
                    Nitioti.append(li[Nit[j]])
                Nitioti = np.array(Nitioti)
                # print(Nitioti,i)
        ## 求荧光素和
                sumNitioti = sum(Nitioti)
        ## 获取萤火虫移动概率公式的分子和分母
                for j in range(num):
                    numerator.append(Nitioti[j] - li[i])
                denominator = sumNitioti - li[i, 0]
        # 移动概率
                for j in range(num):
                    Pij.append(numerator[j] / denominator)
                Pij = np.array(Pij)
        # 概率归一化
                cumsum(Pij)
        ## 确定移动位置
                Pos = []
                for j in range(num):
                    if (random.random() < Pij[j]):
                        pos = j
                        break
                j = Nit[pos]
                x = gAddress1[j, :] - gAddress1[i, :]
                y = gAddress1[i, :] - gAddress[i, :]
                z = x.dot(y)
                # print(z,"z")
                Lx = np.sqrt(x.dot(x))
                Ly = np.sqrt(y.dot(y))
                cta = z / (Lx * Ly+1e-100)
                if (cta < 0):
                    s[i][0] = s[i][0] / (2 - cta)
                else:
                    a = min(s[i][0] * (1 + 0.5 * cta), smax)
                    if (a < smax):
                        s[i][0] = a
                    else:
                        s[i][0] = smax
        #                 ## i点向j点移动
                ADD[i,:] = gAddress1[i, :] + s[i][0] * ((gAddress1[j, :] - gAddress1[i, :]) / norm(gAddress1[j, :], gAddress1[i, :]))
        #                 # ## 限制i点的移动范围
                range1(ADD, domx)
                    ## 更新决策半径
                rdi[i] = rdi[i] + beta * (nt - len(Nit))
                rdi[i] = min(rs, max(0.5, rdi[i]))
        # print(s)
        for i in range(n):
            gAddress[i,:] = gAddress1[i,:]
        for i in range(n):
            for j in range(dim):
                gAddress1[i,j] = ADD[i][j]
    #     # # 获取最优个体
        objectvalue2 = ObjValue(gAddress1)
        best_fit1, index_best1 = best(objectvalue2)
        if (best_fit1 > currentbestfit):
            currentbestfit = best_fit1
        if (best_fit1 > currentbestfit):
            currentbestfit = objectvalue2[index_best1]
        value.append(float(currentbestfit))
        print(generation , "+", currentbestfit,"+",gAddress1[index_best,:])
        # print(gAddress1[index_best,:])
    # print(objectvalue2)
    # print(gAddress1)
    # newpop = []
    ll, = plt.plot(gAddress1[:, 0], gAddress1[:, 1], '+')  # 传hadle需要，
    plt.legend(handles=[ll], labels=['class'], loc='best')
    plt.show()
    # for i in range(len(objectvalue2)):
    #     newpop.append(objectvalue2[i])
    # newpop.sort(reverse=True)
    # # print(newpop)
    # N0 = (int)(a * len(fit_value))
    # print(s)
    x = np.linspace(0, maxGeneration, maxGeneration+1)
    # print(x)
    # print(len(value))
    plt.plot(x, value)
    plt.show()

