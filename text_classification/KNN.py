
import  pickle


# 读取bunch对象
def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


# 导入训练集
trainpath = "./outputData/train_word_bag/tfidfspace.dat"
train_set = _readbunchobj(trainpath)
# 导入测试集
testpath = "./outputData/test_word_bag/testspace.dat"
test_set = _readbunchobj(testpath)

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)  # default with 'rbf'
neigh.fit(train_set.tdm, train_set.label)
# 预测分类结果
predicted = neigh.predict(test_set.tdm)

for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        print(file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate)

print("预测完毕!!!")
# 计算分类精度：
from sklearn import metrics


def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))
metrics_result(test_set.label, predicted)