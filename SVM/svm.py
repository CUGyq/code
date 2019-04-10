from numpy import *
import time
import numpy as np
import matplotlib.pyplot as plt
# calulate kernel value
def calcKernelValue(matrix_x, sample_x, kernelOption):
    kernelType = kernelOption[0]
    numSamples = matrix_x.shape[0]
    kernelValue = mat(zeros((numSamples, 1)))
    if kernelType == 'linear':
        kernelValue = matrix_x * sample_x.T
    elif kernelType == 'rbf':
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(numSamples):
            diff = matrix_x[i, :] - sample_x
            kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma ** 2))
    else:
        raise NameError('Not support kernel type! You can use linear or rbf!')
    return kernelValue
# calculate kernel matrix given train set and kernel type
def calcKernelMatrix(train_x, kernelOption):
    numSamples = train_x.shape[0]
    kernelMatrix = mat(zeros((numSamples, numSamples)))
    for i in range(numSamples):
        kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)
    return kernelMatrix
# define a struct just for storing variables and data
class SVMStruct:
    def __init__(self, dataSet, labels, C, toler, kernelOption):
        self.train_x = dataSet  # each row stands for a sample
        self.train_y = labels  # corresponding label
        self.C = C  # 松弛变量
        self.toler = toler  # 迭代终止条件
        self.numSamples = dataSet.shape[0]  # 训练个数

        self.alphas = mat(zeros((self.numSamples, 1)))  # Lagrange factors for all samples

        self.b = 0
        self.errorCache = mat(zeros((self.numSamples, 2)))
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)


        # calculate the error for alpha k
def calcError(svm, alpha_k):
    output_k = float(multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_k] + svm.b)  #g(x)i
    # error_k =float(svm.train_y[alpha_k])*output_k-1
    error_k = output_k - float(svm.train_y[alpha_k])
    # print(svm.train_y[alpha_k])
    return error_k
# update the error cache for alpha k after optimize alpha k
def updateError(svm, alpha_k):
    error = calcError(svm, alpha_k)
    svm.errorCache[alpha_k] = [1, error]
# select alpha j which has the biggest step
def selectJrand(i,m):
    j = i
    while(i == j):
        j = int(random.uniform(0,m))
    return j
def selectAlpha_j(svm, alpha_i, error_i):
    E_j = 0
    maxK = -1
    maxDeltaE = 0
    svm.errorCache[alpha_i] = [1, error_i]  # mark as valid(has been optimized)
    # print(svm.errorCache)


    candidateAlphaList = nonzero(svm.errorCache[:, 0].A)[0]  # mat.A return array

    # print(svm.errorCache[:, 0])
    # print(candidateAlphaList)
    # find the alpha with max iterative step
    # print(len(candidateAlphaList))
    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_k - error_i) > maxDeltaE:
                maxDeltaE = abs(error_k - error_i)
                maxK = alpha_k
                E_j = error_k

        return maxK ,E_j
                # if came in this loop first time, we select alpha j randomly
    else:
        j = selectJrand(alpha_i,svm.numSamples)

        error_j = calcError(svm, j)

    return j, error_j
# the inner loop for optimizing alpha i and alpha j
def innerLoop(svm, alpha_i):
    error_i = calcError(svm, alpha_i)

    ### check and pick up the alpha who violates the KKT condition
    ## satisfy KKT condition
    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
    ## violate KKT condition
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
    if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or \
                    (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):
        # step 1: select alpha j
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)


        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # step 2: calculate the boundary L and H for alpha j
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            print("L == H")
            return 0

            # step 3: calculate eta (the similarity of sample i and j)
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] \
              - svm.kernelMat[alpha_j, alpha_j]
        if eta >= 0:
            print("eta>=0")
            return 0

            # step 4: update alpha j
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta

        # step 5: clip alpha j
        svm.alphas[alpha_j] = clipAlpha(svm.alphas[alpha_j],H,L)
        updateError(svm, alpha_j)

            # step 6: if alpha j not moving enough, just return
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            print("alpha_j is not moving enogh")
            # updateError(svm, alpha_j)
            return 0

            # step 7: update alpha i after optimizing aipha j
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
                               * (alpha_j_old - svm.alphas[alpha_j])
        updateError(svm, alpha_i)

        # step 8: update threshold b
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                               * svm.kernelMat[alpha_i, alpha_i] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
               * svm.kernelMat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                               * svm.kernelMat[alpha_i, alpha_j] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
               * svm.kernelMat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

            # step 9: update error cache for alpha i, j after optimize alpha i, j and b
        # updateError(svm, alpha_j)
        # updateError(svm, alpha_i)

        return 1
    else:
        return 0


        # the main training procedure
def trainSVM(train_x, train_y, C, toler, maxIter, kernelOption=('linear', 0)):
    # calculate training time
    startTime = time.time()
    # init data struct for svm
    svm = SVMStruct(mat(train_x), mat(train_y), C, toler, kernelOption)
    # start training
    entireSet = True
    alphaPairsChanged = 0
    iterCount = 0
    # Iteration termination condition:
    #   Condition 1: reach max iteration
    #   Condition 2: no alpha changed after going through all samples,
    #                in other words, all alpha (samples) fit KKT condition
    while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        # print(svm.alphas)
        alphaPairsChanged = 0
        # update alphas over all training examples
        if entireSet:
            for i in range(svm.numSamples):

            # for i in range(svm.numSamples):
                alphaPairsChanged += innerLoop(svm, i)
            print( '---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
            # print(svm.alphas)
            iterCount += 1
            # update alphas over examples where alpha is not 0 & not C (not on boundary)
        else:
            nonBoundAlphasList = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]  #找到边界上的点
            # print(nonBoundAlphasList.shape)
            for i in nonBoundAlphasList:
                alphaPairsChanged += innerLoop(svm, i)

            print('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
            iterCount += 1

            # alternate loop over all examples and non-boundary examples
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True

    print('Congratulations, training complete! Took %fs!' % (time.time() - startTime))
    # print(svm.alphas)
    return svm,svm.b,svm.alphas
# testing your trained svm model given test set
def testSVM(svm, test_x, test_y):
    test_x = mat(test_x)
    test_y = mat(test_y)
    numTestSamples = test_x.shape[0]
    supportVectorsIndex = nonzero((svm.alphas.A > 0))[0]
    supportVectors = svm.train_x[supportVectorsIndex]
    # print(supportVectors)
    supportVectorLabels = svm.train_y[supportVectorsIndex]
    supportVectorAlphas = svm.alphas[supportVectorsIndex]
    train_matchCount = 0
    test_matchCount = 0
    # for i in range(numTestSamples):
    #     kernelValue = calcKernelValue(supportVectors, svm.train_x[i, :], svm.kernelOpt)
    #     predict = kernelValue.T * multiply(supportVectorLabels, supportVectorAlphas) + svm.b
    #     # print(predict)
    #     # print(test_y[i])
    #     if sign(predict) == sign(svm.train_y[i]):
    #         train_matchCount += 1
    # train_accuracy = float(train_matchCount) / numTestSamples
    for i in range(numTestSamples):
        kernelValue = calcKernelValue(supportVectors, test_x[i, :], svm.kernelOpt)
        predict = kernelValue.T * multiply(supportVectorLabels, supportVectorAlphas) + svm.b
        # print(predict)
        # print(test_y[i])
        if sign(predict) == sign(test_y[i]):
            test_matchCount += 1
    test_accuracy = float(test_matchCount) / numTestSamples
    return test_accuracy
# show your trained svm model only available with 2-D data
def showSVM(svm):
    # print(svm.train_x.shape)
    if svm.train_x.shape[1] != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

        # draw all samples
    for i in range(svm.numSamples):
        if svm.train_y[i] == -1:
            plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'og')
        elif svm.train_y[i] == 1:
            plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'ob')

# mark support vectors
    # plt.scatter(svm.train_x[:, 0], svm.train_x[:, 1], c=svm.train_y, cmap=plt.cm.Paired)  # [:，0]列切片，第0列
    # a = np.array(svm.train_x)
    # b = list(svm.train_y)
    # plt.scatter(a[:, 0], a[:, 1], c=b, cmap=plt.cm.Paired)  # [:，0]列切片，第0列
    # print(type(a))
    supportVectorsIndex = nonzero((svm.alphas.A > 0) * (svm.alphas.A <= svm.C))[0]
    supportVectors = svm.train_x[supportVectorsIndex]
    # print(supportVectors)
    for i in supportVectorsIndex:
        plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')

        # draw the classify line
    w = zeros((2, 1))
    # print(w.shape)
    for i in range(svm.numSamples):
    # for i in supportVectorsIndex:
        w += multiply(svm.alphas[i] * svm.train_y[i], svm.train_x[i, :].T)
    # print(w[0])
    aa = -w[0] / w[1]  # 斜率
    # print(aa)
    min_x = min(svm.train_x[:, 0])[0,0]
    # print(min_x.shape)
    max_x = max(svm.train_x[:, 0])[0, 0]
    # y_min_x = float(aa*min_x + svm.b)
    # y_max_x = float(aa*max_x+svm.b)
    y_min_x = float(-svm.b - w[0] * min_x) / w[1]
    y_max_x = float(-svm.b - w[0] * max_x) / w[1]
    l = supportVectors[0].A
    # print(l.shape)
    yy_down_min = aa * min_x + (l[0,1] - aa * l[0,0])
    yy_down_max = aa * max_x + (l[0,1] - aa *l[0,0])
    l = supportVectors[-1].A
    # print(l.shape)
    yy_up_min = aa * min_x + (l[0, 1] - aa * l[0, 0])
    yy_up_max = aa * max_x + (l[0, 1] - aa * l[0, 0])
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-r')
    plt.plot([min_x, max_x], [yy_down_min, yy_down_max], '-g')
    plt.plot([min_x, max_x], [yy_up_min, yy_up_max], '-g')
    plt.show()
#load dataSet
def loadDataSet(fileName):
    dataSet = []
    labels = []
    fileIn = open(fileName)
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
        labels.append(float(lineArr[2]))
    dataSet = mat(dataSet)
    labels = mat(labels).T
    return dataSet , labels
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj