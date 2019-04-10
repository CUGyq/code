
import cv2
img = cv2.imread('./test/1.jpg')
cv2.imshow("a",img)
cv2.waitKey(0)
import numpy as np
from pylab import *
import glob
import os
i = 0

img = cv2.imread('./test/1.jpg')
img = cv2.resize(img, (300, 300), 0)
h, w = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayIPimage = gray.copy()

x = cv2.Sobel(grayIPimage,cv2.CV_16S,1,0)
absX = cv2.convertScaleAbs(x)   # 转回uint8


ret,thresh1=cv2.threshold(absX,127,255,cv2.THRESH_OTSU)
cv2.imshow("a",thresh1)
cv2.waitKey(0)



kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))

dilated = cv2.dilate(img,kernel)
eroded = cv2.erode(dilated,kernel)
dilated = cv2.dilate(eroded,kernel)


#     kernal = cv.CreateStructuringElementEx(1, 3, 0, 1, 0)
#     cv.Erode(temp, temp, kernal, 1)
#     cv.Dilate(temp, temp, kernal, 3)
#     #     cv.ShowImage('2', temp)
#     temp = np.asarray(cv.GetMat(temp))
#     contours, heirs = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     for tours in contours:
#         rc = cv2.boundingRect(tours)
#         if rc[2] / rc[3] >= 2:
#             # rc[0] 表示图像左上角的纵坐标，rc[1] 表示图像左上角的横坐标，rc[2] 表示图像的宽度，rc[3] 表示图像的高度，
#             cv2.rectangle(image, (rc[0], rc[1]), (rc[0] + rc[2], rc[1] + rc[3]), (255, 0, 255))
#             imageIp = cv.GetImage(cv.fromarray(image))
#             cv.SetImageROI(imageIp, rc)
#             imageCopy = cv.CreateImage((rc[2], rc[3]), cv2.IPL_DEPTH_8U, 3)
#             cv.Copy(imageIp, imageCopy)
#             cv.ResetImageROI(imageIp)
#             cv.SaveImage('D:/pic/result/' + str(i) + '.jpg', imageCopy)
#             i = i + 1
# # cv2.imshow("黑底白字",image)
# cv2.waitKey(0)  # 暂停用于显示图片