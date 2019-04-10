from numpy import *
import svm
from sklearn.externals import joblib
################## test svm #####################
## step 1: load data
print("step 1: load data...")
# dataSet,labels = svm.loadDataSet('testSet.txt')
# train_x = dataSet[0:161, :]
# train_y = labels[0:161, :]
# test_x = dataSet[160:201, :]
# test_y = labels[160:201, :]
dataSet = []
labels = []
test_x = []
test_y = []
fileIn = open('./titanic（3维）/titanic_test_data.asc')
for line in fileIn.readlines():
    # print(line)
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
    # print(dataSet)
    # labels.append(float(lineArr[2]))
filelabel = open('./titanic（3维）/titanic_test_label.asc')
for line in filelabel.readlines():
    lineArr = line.strip().split('\t')
    labels.append(float(lineArr[0]))
train_x = mat(dataSet)
# print(dataSet)
train_y = mat(labels).T


fileIn = open('./titanic（3维）/titanic_train_data.asc')
for line in fileIn.readlines():
    # print(line)
    lineArr = line.strip().split('\t')
    test_x.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
    # print(dataSet)
    # labels.append(float(lineArr[2]))
filelabel = open('./titanic（3维）/titanic_train_label.asc')
for line in filelabel.readlines():
    lineArr = line.strip().split('\t')
    test_y.append(float(lineArr[0]))
train_x = mat(dataSet)
# print(dataSet)
train_y = mat(labels).T
test_x = mat(test_x)
# print(dataSet)
test_y = mat(test_y).T
# # ## step 2: training...
print("step 2: training...")
C = 0.6
toler = 0.001
maxIter = 40
svmClassifier,b , alpha  = svm.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption=('linear', 0))
# print(b)
# print(alpha[alpha>0])
joblib.dump(svmClassifier, "train_model.m")#保存模型
# svmClassifier = joblib.load("train_model.m")#加载模型
# # ## step 3: testing
print("step 3: testing...")
test_accuracy = svm.testSVM(svmClassifier, test_x, test_y)
# print(accuracy)
print('The classify test_accuracy is: %.3f%%' % (test_accuracy * 100))
## step 4: show the result
print("step 4: show the result...")
svm.showSVM(svmClassifier)
#


