import numpy as np
import matplotlib.pyplot as plt
import math
import random
def GenarationInitialPopulation(pop_size, chrom_length):
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)
    return pop[1:]
def decodechrom(pop,k,length):
    temp = []
        # i = 0
    for i in range(len(pop)):
        # print(i)
        t = 0
        for j in range(k,length):
    #         # print(j)
            t += pop[i][j] * (math.pow(2, j-k))
        # print(type(t))
        temp.append(t)
        # print(temp)
    return temp
def calfitValue(obj_value):
    fit_value = []
    c_min = 0
    for i in range(len(obj_value)):
        if(obj_value[i] + c_min > 0):
            temp = c_min + obj_value[i]
        else:
            temp = 0.0
        fit_value.append(temp)
    return fit_value
def calobjValue(pop):
    # temp = [[]]
    obj_value = []
    temp1 = decodechrom(pop,0,LENGTH1)
    temp2 = decodechrom(pop,LENGTH1,CHROMLENGTH)
    # temp.append(temp1)
    # temp.append(temp2)
    for i in range(len(pop)):
        x = ((4.096*temp1[i])/1023)-2.048
        y = ((4.096 * temp2[i]) / 1023) - 2.048
        a = math.pow(x,2)-y
        b = math.pow(a,2)
        c = 1-x
        f = 100*b+math.pow(c,2)
        # z = f
        obj_value.append(f)
    return obj_value
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
def selection(pop, fit_value):
    newfit_value = []
    # 适应度总和
    total_fit = sum(fit_value)
    # print(total_fit)
    for i in range(len(fit_value)):
        newfit_value.append(fit_value[i] / total_fit)
    # print(newfit_value)
        # 计算累计概率
    cumsum(newfit_value)
    # print(newfit_value)
    newpop = pop
    # 转轮盘选择法
    for i in range(len(pop)):
        index = 0
        p = random.random()
        if (p > newfit_value[index]):
            index = index + 1
        else:
             newpop[i] = pop[index]
    pop = newpop
    return pop
def crossover(pop, pc):
    pop_len = len(pop)
    # print(len(pop[0]))
    # print(pop[0],"a")
    for i in range(pop_len - 1):
        # print(i)
        if(random.random() < pc):
            cpoint = random.randint(0,len(pop[0]))
            # print(type(cpoint))
            # print(type(pop[i][0:cpoint]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i+1][cpoint:len(pop[i])])
            temp2.extend(pop[i+1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1
            pop[i+1] = temp2
    return pop
def best(pop, fit_value):
    px = len(pop)
    index_worst = 0
    index_best = 0
    currentbestfit = []
    currentbestindividual = []
    best_individual = []
    best_fit = fit_value[0]
    worst_individual = []
    worst_fit = fit_value[0]
    for i in range(1, px):
        if(fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
            index_best =i
        if(fit_value[i]<worst_fit):
            worst_fit = fit_value[i]
            worst_individual = pop[i]
            index_worst = i
    return [best_fit,best_individual,index_best,worst_fit,worst_individual,index_worst]
def PerformEvolution(p):
     if(p>best_currentfit):
         best_currentfit = p
     else:
         pass
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])
    # X = []
    # Y = []
    # for i in range(px):
    #     X.append(pop[i])
    #
    for i in range(px):
        # X.append(pop[i])
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1
    return pop
if __name__ == '__main__':
    genaration = 0
    MaxGeneration = 500
    PopSize = 80
    LENGTH1 = 10
    LENGTH2 = 10
    pc = 0.6
    pm = 0.001
    CHROMLENGTH = LENGTH1+LENGTH2
    chrom = GenarationInitialPopulation(PopSize,CHROMLENGTH )
    value = calobjValue(chrom)
    fit_value = calfitValue(value)
    best_fit,best_individual,index_best, worst_fit,worst_individual, index_worst= best(chrom, fit_value)
    currentbestfit = best_fit
    currentbestindividual = best_individual
    while(genaration<MaxGeneration):
        genaration+=1
        chrom1 = selection(chrom, fit_value)
        chrom2 = crossover(chrom1, pc)
        chrom3 = mutation(chrom2, pm)
        value1 = calobjValue(chrom3)
        fit_value1 = calfitValue(value1)
        best_fit1, best_individual1, index_best1, worst_fit1, worst_individual1, index_worst1 = best(chrom3, fit_value1)
        if(best_fit1>currentbestfit):
            currentbestfit = best_fit1
            currentbestindividual = best_individual1
        if(best_fit1>currentbestfit):
            currentbestfit = fit_value1[index_best1]
            currentbestindividual = chrom3[index_best1]
        else:
            fit_value1[index_worst1] = currentbestfit
            chrom3[index_worst1] = currentbestindividual
        print(genaration,"+",currentbestfit,currentbestindividual)
