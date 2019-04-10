import numpy as np
def loadDataSet(): #导入数据
    #假设数据为最简单的6篇文章，每篇文章大概7~8个词汇左右，如下
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #对应上述6篇文章的分类结果，1为侮辱性，0为非侮辱性
    return postingList,classVec
def createVocabList(dataSet):# 将所有文章中的词汇取并集汇总
    vocabSet = set([])  # 定义一个set(set存储的内容无重复)
    for document in dataSet:# 遍历导入的dataset数据，将所有词汇取并集存储至vocabSet中
        vocabSet = vocabSet | set(document) # | 符号为取并集，即获得所有文章的词汇表
    return list(vocabSet)
#该函数输入参数为词汇表及某篇文章，输出为文档向量，向量每个元素为1或0，分别表示词汇表中的单词在输入文档中是否出现；
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList) #构建一个0向量；

    for word in inputSet: # 遍历词汇表，如果文档中出现了词汇表中单词，则将输出的文档向量中对应值设为1，旨在计算各词汇出现的次数；
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1#因为上一段代码里，给的文章例子里的单词都是不重复的，如果有重复的单词的话，这段代码改写为：returnVec[vocabList.index(word)] += 1更为合适；
        else:
            print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec#返回向量化后的某篇文章
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #计算有多少篇文章
    numWords = len(trainMatrix[0]) #计算第一篇文档的词汇数
    pAbusive = sum(trainCategory)/float(numTrainDocs) #计算p(c_1)，p(c_0)=1-p(c_1)

    p0Num = np.ones(numWords) #构建一个空矩阵，用来计算非侮辱性文章中词汇数

    p1Num = np.ones(numWords) #构建一个空矩阵，用来计算侮辱性文章中词汇数

    p0Denom = 2.0; p1Denom = 2.0

    for i in range(numTrainDocs): #遍历每一篇文章，来求P(w|c)
        if trainCategory[i] == 1: #判断该文章是否为侮辱性文章
            p1Num += trainMatrix[i] #累计每个词汇出现的次数
            p1Denom += sum(trainMatrix[i]) #计算所有该类别下不同词汇出现的总次数
        else:#如果该文章为非侮辱性文章
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom) #计算每个词汇出现的概率P(w_i|c_1)
    p0Vect = np.log(p0Num/p0Denom) #计算每个词汇出现的概率P(w_i|c_0)
    return p0Vect,p1Vect,pAbusive
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):#vec2Classify为输入的一个向量化的新文章
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)# 由于是取对数，所以乘积变为直接求和即可，注意这里list和list是无法相乘，vec2Classify需要为array格式
    p0 = sum(vec2Classify * p0Vec) + np.log(1-pClass1)
    if(p1>p0):
        return 1
    if(p0>p1):
        return 0
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    # testEntry = ['stupid', 'garbage']
    # thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    # print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
if __name__ == '__main__':
    testingNB()
    # listOPosts,listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    # # 将所有文章转化为向量，并将所有文章中的词汇append到traninMat中
    # p0V,p1V,pAb = trainNB0(trainMat,listClasses)# 计算训练集的p(c_1),p(w_i|c_0),p(w_i|c_1)
