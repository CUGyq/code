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
    for i in range(len(obj_value)):
        v = -(r*obj_value[i])
        fit_value.append( 1.0 / (math.e**v+ 1.0))
    # print(fit_value)
    return fit_value
def calobjValue(pop):
    obj_value = []
    temp1 = decodechrom(pop,0,LENGTH1)
    temp2 = decodechrom(pop,LENGTH1,CHROMLENGTH)
    for i in range(len(pop)):
        x = ((4.096*temp1[i])/1023)-2.048
        y = ((4.096 * temp2[i]) / 1023) - 2.048
        a = math.pow(x,2)-y
        b = math.pow(a,2)
        c = 1-x
        f = 100*b+math.pow(c,2)
        obj_value.append(f)
    return obj_value
def sum1(fit_value):
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
    X = []
    Y = []
    newpop = []

    for i in range(len(fit_value)):
        X.append(fit_value[i])

    for j in range(len(X) - 1):
        for i in range(len(X) - j - 1):
            if (X[i] < X[i + 1]):
                X[i],X[i + 1] = X[i + 1], X[i]
    # X.sort(reverse=True)

    N0 =round(a*len(fit_value))
    # print(N0)
    for i in range(N0):
        Y.append(X[i])
    # print(Y)
    # print(len(Y))
    for i in range(N0):
        for j in range(len(fit_value)):
            if(Y[i] == fit_value[j]):
                newpop.append(pop[j])
    return [newpop,Y]
def best(value,pop, fit_value):
    px = len(pop)
    index_worst = []
    index_best = []
    best_individual = pop[0]
    best_fit = fit_value[0]
    best_value = value[0]
    worst_individual = pop[0]
    worst_fit = fit_value[0]
    worst_value = value[0]
    for i in range(1, px):
        if(fit_value[i] > best_fit):
        # if (value[i] > best_value):
            best_fit = fit_value[i]
            best_individual = pop[i]
            best_value = value[i]
            index_best = i
        if(fit_value[i]<worst_fit):
        # if (value[i] < worst_value):
            worst_fit = fit_value[i]
            worst_individual = pop[i]
            worst_value = value[i]
            index_worst = i
    return [best_fit,best_individual,best_value,index_best,worst_fit,worst_individual,worst_value,index_worst]
def mutation(pop,fit_value):
    newpop = np.array(pop)
    px = len(newpop)
    py = len(newpop[0])
    pc = []
    for i in range(px):
        v = (-1)*fit_value[i]
        pc.append(math.e**v)
    for i in range(px):
        for j in range(py):
            if(random.random()<pc[i]):
                if (newpop[i][j] == 1):
                    newpop[i][j] = 0
                else:
                    newpop[i][j] = 1
    # for i in range(px):
    #     if(random.random()<pc[i]):
    #         mpoint = random.randint(0, py - 1)
    #         if (newpop[i][mpoint] == 1):
    #             newpop[i][mpoint] = 0
    #         else:
    #             newpop[i][mpoint] = 1
    # for i in range(len(fit_value)):
    #     print(pop[i])
    #     print(newpop[i])
    return newpop
def clonenum(pop,fit_value):
    # M = 120
    # m = []
    sumfit = []
    newfit_value = []
    newpop = []
    sumfit = sum1(fit_value)
    # for i in range(len(fit_value)):
    #     a = fit_value[i] * M / sumfit
    #     m.append(round(a))
    for j in range(len(fit_value)):
        for k in range(50):
        # for k in range(m[j]):
            newfit_value.append(fit_value[j])
            newpop.append(pop[j])
    # N1 = M - len(newpop)
    # if(N1<0):
    #     pass
    # else:
    #     # print(N1)
    #     Y = []
    #     for i in range(N1):
    #         X = []
    #         for j in range(CHROMLENGTH):
    #             X.append(random.randint(0, 1))
    #         Y.append(X)
    #         m.append(1)
    #     v = calobjValue(Y)
    #     fit = calfitValue(v)
    #     for j in range(N1):
    #         newpop.append(Y[j])
    #         newfit_value.append(fit[j])
    return newfit_value,newpop
def CalculateSimilarity(X1,X2):
    num = 0.0
    for i in range(CHROMLENGTH):
        if (X1[i] == X2[i]):
            num += 0
        else:
            num+=1
    a  = num * (math.log(2.0)) / CHROMLENGTH
    return a
def Inhibition(pop):
    value = calobjValue(pop)
    fit_value = calfitValue(value)
    # X=[]
    # Y=[]
    # for i in range(len(fit_value)):
    #     X.append(fit_value[i])
    # for j in range(len(fit_value)-1):
    #     for i in range(len(fit_value)-j-1):
    #         if(fit_value[i]<fit_value[i+1]):
    #             fit_value[i],fit_value[i+1] = fit_value[i+1],fit_value[i]
    #             pop[i],pop[i+1] = pop[i+1],pop[i]
    # print(fit_value)
    newpop = []
    newfit_value = []
    newpop.append(pop[0])
    newfit_value.append(fit_value[0])
    # L= 0
    # for i in range(len(pop)-1):
    #     for j in range(i+1,len(pop)):
    #         a = CalculateSimilarity(pop[i], pop[j])
    #         if(a>rod):
    #             L+=1
    #             pop[L]=pop[j]
    #             fit_value[L]=fit_value[j]
    #     L=i+1

    # k = 0
    # # newpop.append(pop[0])
    # # newfit_value.append(fit_value[0])
    # for i in range(len(pop)):
    #     print(k)
    #     for j in range(1,m[k]):
    #         a = CalculateSimilarity(pop[i], pop[i+j])
    #         if (a < rod):
    #             pass
    #         else:
    #             newpop.append(pop[i+j])
    #             newfit_value.append(fit_value[i+j])
    #     i +=m[k]
    #     k +=1



    for i in range(1,len(pop)):
        for j in range(len(newpop)):
            a = CalculateSimilarity(pop[i], newpop[j])
            b = np.array(a)
            # print(b)
            if (b < rod):
                if (newfit_value[j] < fit_value[i]):
                    newpop[j] = pop[i]
                    newfit_value[j] = fit_value[i]
                    # print(newfit_value[j])
                    break
                else:
                    break
            else:
                if(j == (len(newpop)-1)):
                    newpop.append(pop[i])
                    newfit_value.append(fit_value[i])
    return pop,fit_value
def chongzu(pop1,pop2,fit_value1,fit_value2):
    newpop = []
    newfit_value = []
    px = len(pop1)
    py = len(pop2)
    for i in range(px):
        newpop.append(pop1[i])
        newfit_value.append(fit_value1[i])
    for j in range(py):
        newpop.append(list(pop2[j]))
        newfit_value.append(fit_value2[j])
    for j in range(len(newfit_value) - 1):
        for i in range(len(newfit_value) - j - 1):
            if (newfit_value[i] < newfit_value[i + 1]):
                newfit_value[i], newfit_value[i + 1] = newfit_value[i + 1], newfit_value[i]
                newpop[i], newpop[i + 1] =newpop[i + 1], newpop[i]
    # print(len(newfit_value))
    return newpop,newfit_value
def CalculateConcentrationValue(pop):
    px = len(pop)
    # print(px)
    num = 0
    C = []
    for i in range(px):
        # print(i)
        for j in range(px):
           a = CalculateSimilarity(pop[i],pop[j])
           if (a<rod):
              num +=1
        # print(num)
        C.append((num*(1.0))/px)
        # print(num)
        num = 0
    return C
def CalculateActivityValue(fit_value,c):
    bet = 1.5
    px = len(fit_value)
    act = []
    for i in range(px):
        a = -(c[i]/bet)
        b = math.e**a
        act.append(fit_value[i]*b)
    return act
def activeslect(pop,fit_value):
    X = []
    Y = []
    newpop = []
    for i in range(len(fit_value)):
        X.append(fit_value[i])
    # # X.sort(reverse=True)
    for i in range(PopSize):
        Y.append(X[i])
        newpop.append(pop[i])
    # # for i in range(PopSize):
    # #     for j in range(len(fit_value)):
    # #         if(Y[i] == fit_value[j]):
    # #             newpop.append(pop[j])
    # for i in range(len(PopSize)):
    #
    # print(Y)
    acttn = []
    ACT = []
    Tn = math.log((MaxGeneration * (1.0) / genaration) + 1)
    N3 = int(PopSize * u)
    # print(N3)
    c = CalculateConcentrationValue(newpop)
    act = CalculateActivityValue(Y,c)
    for i in  range(PopSize):
        acttn.append(math.e**((act[i]) / Tn))
    total_act = sum(acttn)
    for i in range(PopSize):
        ACT.append(acttn[i] / total_act)
    cumsum(ACT)
    newpop1 = newpop
    for i in range(PopSize):
        index = 0
        p = random.random()
        if (p > ACT[index]):
            index += 1
        else:
             newpop1[i] = newpop[index]
    resultpop = []
    for i in range(PopSize-N3):
        resultpop.append(newpop1[i])
    return resultpop
def sortnewmember(pop):
    newpop = []
    N3 = (int)(PopSize * u)
    # print(len(pop))
    # print(N3)
    Y = []
    for i in range(N3):
        X = []
        for j in range(CHROMLENGTH):
            X.append(random.randint(0, 1))
        Y.append(X)
    px = len(pop)
    for i in range(px):
        newpop.append(pop[i])
    for j in range(N3):
        newpop.append(Y[j])
    # print(len(newpop))
    return newpop
if __name__ == '__main__':
    genaration = 0
    MaxGeneration = 200
    PopSize = 80
    LENGTH1 = 10
    LENGTH2 = 10
    POPSIZE = 300
    r = 0.001#亲和力参数
    a = 0.6 #克隆选择率
    rod = 0.30#抑制半径
    u = 0.06
    CHROMLENGTH = LENGTH1+LENGTH2
    chrom = GenarationInitialPopulation(PopSize,CHROMLENGTH )
    value = calobjValue(chrom)
    fit_value = calfitValue(value)
    best_fit, best_individual, best_value, index_best, worst_fit, worst_individual, worst_value, index_worst = best(value ,chrom, fit_value)
    currentbestfit = best_fit
    currentbestindividual = best_individual
    currentbestvalue = best_value
    value = []
    value.append(currentbestvalue)
    # l0, = plt.plot()
    while(genaration<MaxGeneration):
        genaration+=1
        chrom1,fit_value1 = selection(chrom, fit_value)
        fit_value2, chrom2 = clonenum(chrom1,fit_value1)
        chrom3= mutation(chrom2,fit_value2)
        chrom4,fit_value4 = Inhibition(chrom3)
        chrom5,fit_value5 = chongzu(chrom, chrom4, fit_value, fit_value4)
        chrom6 = activeslect(chrom5, fit_value5)
        chrom7 = sortnewmember(chrom6)
        value1 = calobjValue(chrom7)
        fit_value6 = calfitValue(value1)
        best_fit1, best_individual1, best_value1, index_best1, worst_fit1, worst_individual1, worst_value1, index_worst1 = best(value1, chrom7, fit_value6)
        if (best_fit1 > currentbestfit):
            currentbestfit = best_fit1
            currentbestindividual = best_individual1
            currentbestvalue = best_value1
        if (best_fit1 > currentbestfit):
            currentbestfit = fit_value6[index_best1]
            currentbestindividual = chrom7[index_best1]
            currentbestvalue = value1[index_best1]
        else:
            fit_value6[index_worst1] = currentbestfit
            chrom7[index_worst1] = currentbestindividual
            value1[index_worst1] = currentbestvalue
        value.append(currentbestvalue)
        print("第",genaration,"代", "+","最优值为",currentbestvalue,"坐标为",currentbestindividual)
    # print(len(value))
    x = np.linspace(0,MaxGeneration,MaxGeneration+1)
    plt.plot(x,value)
    plt.show()