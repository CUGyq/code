import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cholesky
from numpy import random
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
# 获取适应度值
def ObjValue(h):
    objvalue = zeros((n,1))
    for i in range(n):
        # 函数1：
        objvalue[i,0] = -(math.log2((1/n)))+math.log2(h[i])
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

    for i in range(k):
        for j in range(len(t[0])):
            if(t[i][j]<Y[i][0]):
                t[i][j] = Y[i][0]
            if(t[i][j]>Y[i][1]):
                t[i][j] = Y[i][1]
    for i in range(k):
        X[:,i] = t[i,:]
if __name__ == '__main__':
    sampleNo = 50
    mu0 = np.array([[0, 0]])
    mu1 = np.array([[3, 3]])
    mu2 = np.array([[-3, -3]])
    mu3 = np.array([[3, -3]])
    Sigma = np.array([[1, 0.1], [0.1, 1]])
    R = cholesky(Sigma)
    np.random.seed(0)
    s0 = np.dot(np.random.randn(sampleNo, 2), R) + mu0
    s1 = np.dot(np.random.randn(sampleNo, 2), R) + mu1
    s2 = np.dot(np.random.randn(sampleNo, 2), R) + mu2
    s3 = np.dot(np.random.randn(sampleNo, 2), R) + mu3
    # T=np.arctan2(Y,X)#颜色
    plt.figure()
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.xlabel("x")
    plt.ylabel("y")
    l0, = plt.plot(s0[:, 0], s0[:, 1], '+')  # 传hadle需要，
    l1, = plt.plot(s1[:, 0], s1[:, 1], '+')
    l2, = plt.plot(s2[:, 0], s2[:, 1], '+')  # 传hadle需要，
    l3, = plt.plot(s3[:, 0], s3[:, 1], '+')
    plt.legend(handles=[l0, l1, l2, l3], labels=['one', 'two', 'three', 'four'], loc='best')  # loc = upper right best
    plt.show()
    s = np.vstack((s0, s1, s2, s3))
    n, m = s.shape
    gAdd = s
    l, = plt.plot(s[:, 0], s[:, 1],'+')  # 传hadle需要，
    plt.legend(handles=[l], labels=['class1'], loc='best')
    plt.show()
    generation = 0  # 迭代次数
    # # 初始化参数
    # domx = [[-2.048,2.048],[-2.048,2.048]]      # 解空间
    rho = 0.4                   # 荧光素挥发因子
    gamma = 0.6                 # 适应度提取比例
    beta = 0.08                 # 领域变化率
    nt = 5                      # 邻居阈值(邻居萤火虫数)
    s = 0.1                    # 移动步长
    l0 = 5                      # 初始荧光素值
    rs = 5
    r0 = 3
    maxGeneration = 80         # 最大迭代次数
    r = 1
    k = 4

    # # # 分配空间
    li = np.zeros((n,1))           # 存放荧光素值空间
    rdi = np.zeros((n,1))          # 存放决策半径空间

    # ## 初始荧光素值
    li[:,0] = l0
    # ## 初始决策半径
    rdi[:,0] = r0
    # # ## 迭代开始
    while (generation < maxGeneration):
        generation += 1
        H = []
        for i in range(n):
            LN = []

            # num = []
            for j in range(n):
                if (norm(gAdd[i,:],gAdd[j,:])<r):
                    LN.append(j)
            num = len(LN)
            H.append(num/n)
        # print(type(H))
        objvalue = ObjValue(H)
        # print(a)
        for i in range(n):
            li[i, 0] = (1 - rho) * li[i, 0] + gamma * objvalue[i, 0]
    #     ## 各萤火虫开始移动
        for i in range(n):
            Nit = []
            ## 存放萤火虫序号
            for j in range(n):
                if((norm(gAdd[j,:],gAdd[i,:]) < rdi[i]) and (li[i,0] < li[j,0])):
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
                gAdd[i,:] = gAdd[i,:] + s * ((gAdd[j,:] - gAdd[i,:]) / norm(gAdd[j,:],gAdd[i,:])+1e-100)
            else:
                gAdd[i,:] = gAdd[i,:]
            ## 更新决策半径
            rdi[i] = rdi[i] + beta * (nt - len(Nit))
            rdi[i] = min(rs,max(0,rdi[i]))

    # print(gAdd)
    ll, = plt.plot(gAdd[:, 0], gAdd[:, 1], '+')  # 传hadle需要，
    plt.legend(handles=[ll], labels=['class2'], loc='best')
    plt.show()

    # print(x)