import sys
import os  # python内置的包，用于进行文件目录操作，我们将会用到os.listdir函数
import pickle
from sklearn.datasets.base import Bunch
def _readfile(path):
    with open(path, "rb") as fp:  # with as句法前面的代码已经多次介绍过，今后不再注释
        content = fp.read()
    return content
def corpus2Bunch(wordbag_path, seg_path):
    catelist = os.listdir(seg_path)  # 获取seg_path下的所有子目录，也就是分类信息
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)
    # 获取每个目录下所有的文件
    for mydir in catelist:
        class_path = os.path.join(seg_path,mydir)  # 拼出分类子目录的路径
        file_list = os.listdir(class_path)  # 获取class_path下的所有文件
        for file_path in file_list:  # 遍历类别目录下文件
            fullname = os.path.join(class_path,file_path)  # 拼出文件名全路径
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(_readfile(fullname))  # 读取文件内容
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("构建文本对象结束！！！")
if __name__ == "__main__":  # 这个语句前面的代码已经介绍过，今后不再注释
    # 对训练集进行Bunch化操作：
    wordbag_path = "./outputData/train_word_bag/train_set.dat"  # Bunch存储路径，程序输出
    if not os.path.exists("./outputData/train_word_bag"):
        os.makedirs("./outputData/train_word_bag")
    seg_path = "./outputData/train"  # 分词后分类语料库路径，程序输入
    corpus2Bunch(wordbag_path, seg_path)
    # 对测试集进行Bunch化操作：
    wordbag_path = "./outputData/test_word_bag/test_set.dat"  # Bunch存储路径，程序输出
    if not os.path.exists("./outputData/test_word_bag"):
        os.makedirs("./outputData/test_word_bag")
    seg_path = "./outputData/test"  # 分词后分类语料库路径，程序输入
    corpus2Bunch(wordbag_path, seg_path)