import os
import cv2
path = "c/"

img = os.listdir(path)
for name in img:
    n = name.split(".")[0]

    imgpath = path + name
    m = cv2.imread(imgpath)
    r = cv2.resize(m,(650,1000))
    cv2.imwrite("./d/" + n + '.jpg', r)  # 存储为图像
