from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import codecs
import svm
from sklearn.externals import joblib
from sklearn.utils import shuffle
def loadDataSet(fileName):
    dataSet = []
    labels = []
    fileIn = codecs.open(fileName,'r','utf-8')
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
        labels.append(float(lineArr[2]))
    data = np.array(dataSet)
    label = np.reshape(np.array(labels),(len(labels),1))
    temp = np.hstack([data,label])
    np.random.shuffle(temp)
    x = temp[:,:-1]
    y = temp[:,-1]
    x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size=0.2,random_state=22)
    y_train = y_train[:,np.newaxis]
    y_valid = y_valid[:,np.newaxis]
    return x_train, x_valid, y_train, y_valid
def main():
    iris = datasets.load_iris()
    data = iris['data']
    label = iris['target']
    label = np.reshape(label, (label.shape[0], 1))
    datas = np.hstack([data, label])
    np.random.shuffle(datas)
    # datas = shuffle(datas)
    train, test = train_test_split(datas, test_size=0.2, random_state=0)
    # print(train)
    print(test.shape)
    # datas = np.concatenate([data,np.reshape(label,(label.shape[0],1))])
    # print(type(data))
    # print(label.shape)
    # print(datas.shape)

if __name__ == '__main__':
    # data,label = svm.loadDataSet("testSet.txt")
    print("step 1:load data...")
    train,test,train_label,test_label = loadDataSet("testSet.txt")
    print("step 2: training...")
    C = 0.6
    toler = 0.001
    maxIter = 40
    svmClassifier, b, alpha = svm.trainSVM(train, train_label, C, toler, maxIter, kernelOption=('linear', 0))
    # # print(b)
    # # print(alpha[alpha>0])
    joblib.dump(svmClassifier, "train_model.m")  # 保存模型
    # svmClassifier = joblib.load("train_model.m")#加载模型
    # # ## step 3: testing
    print("step 3: testing...")
    test_accuracy = svm.testSVM(svmClassifier, test, test_label)
    # print(accuracy)
    print('The classify test_accuracy is: %.3f%%' % (test_accuracy * 100))
    ## step 4: show the result
    print("step 4: show the result...")
    svm.showSVM(svmClassifier)


