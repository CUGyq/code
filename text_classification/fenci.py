import os
import jieba
import codecs
jieba.load_userdict("userdict.txt")
class Fenci:
    def __init__(self,filePath,stopWordPath,outpath):
        self.path = filePath
        self.stopwordPath = stopWordPath
        self.outpath = outpath
    def creadstoplist(self,stopwordspath):
        stwlist = [line.strip()
                   for line in codecs.open(stopwordspath, 'r', encoding='utf-8').readlines()]
        return stwlist
    def fenci(self):
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        for file in os.listdir(self.path):
            filePath = os.path.join(self.path,file)

            for f in os.listdir(filePath):
                outfile = os.path.join(self.outpath,file)

                if not os.path.exists(outfile):
                    os.makedirs(outfile)
                outfile = codecs.open(outfile +"/" + f,"w",encoding='utf-8')
                fopen = codecs.open(filePath + "/" + f,'rb')

                for eachLine in fopen:
                    words = jieba.cut(eachLine.strip())
                    stwlist = self.creadstoplist(self.stopwordPath)  # 这里加载停用词的路径
                    outstr = ''
                    for word in words:
                        if word not in stwlist:
                            if len(word) > 1:  # 去掉长度小于1的词
                                if word != '\t':
                                    outstr += word
                                    outstr += " "
                    print(outstr)
                    outfile.write(outstr)
                fopen.close()
                outfile.close()

if __name__ == '__main__':
    filePath = "F:/code/python/text/data/corpus/corpus/answer"
    stopWordPath = "stopwords.txt"
    outpath = "outputData/test"
    FC = Fenci(filePath,stopWordPath,outpath)
    FC.fenci()

