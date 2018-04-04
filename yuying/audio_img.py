import wave
import struct
from scipy import *
from pylab import *
import os
def change(path):
    filename = os.listdir(path)
    filename.sort()
    filename.sort(key=lambda x: int(x[5:-4]))
    for i in filename:
        name = i.split('.')[0]
        imgPath = path + i
        wavefile = wave.open(imgPath, 'r')  # open for writing
        nchannels = wavefile.getnchannels()
        sample_width = wavefile.getsampwidth()
        framerate = wavefile.getframerate()
        numframes = wavefile.getnframes()
        print("channel", nchannels)
        print("sample_width", sample_width)
        print("framerate", framerate)
        print("numframes", numframes)
        # 建一个y的数列，用来保存后面读的每个frame的amplitude。
        y = zeros(numframes)
        # for循环，readframe(1)每次读一个frame，取其前两位，是左声道的信息。右声道就是后两位啦。
        # unpack是struct里的一个函数，用法详见http://docs.python.org/library/struct.html。简单说来就是把＃packed的string转换成原来的数据，无论是什么样的数据都返回一个tuple。这里返回的是长度为一的一个
        # tuple，所以我们取它的第零位。
        for i in range(numframes):
            val = wavefile.readframes(1)
            # left = val[0:2]
            # right = val[2:4]
            left_right = val[0:4]
            v = struct.unpack('h', left_right)[0]
            y[i] = v
        # framerate就是44100，文件初读取的值。然后本程序最关键的一步！specgram！实在太简单了。。。
        Fs = framerate
        specgram(y, NFFT=1024, Fs=Fs, noverlap=900)
        resultPath = "./test/" + name
        savefig(resultPath)
        close()
if __name__ == '__main__':
    path = "./train/"
    change(path)