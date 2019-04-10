import cv2
import dlib
import os
import numpy as np
import csv
import pandas as pd
import time
def get_feature():
    predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    detector = dlib.get_frontal_face_detector()
    return predictor,face_rec_model,detector
def get_face_feature(input,output,width = 64,heigth = 64):
    if not os.path.exists(input):
        print("输入路径错误！")
    if not os.path.exists(output):
        os.mkdir(output)
    predictor, face_rec_model, detector = get_feature()
    j = 0
    face_csv = output + "/" + "csv" + "/" + "feature.csv"
    with open(face_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for img in os.listdir(input):
            j += 1
            img_path = os.path.join(input, img)
            print(img_path)
            image = cv2.imread(img_path, 1)
            dets = detector(image, 1)
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = image[x1:y1, x2:y2]
                cv2.rectangle(image, (x2, x1), (y2, y1), (0, 255, 0), 1)
                # print("正在读的人脸图像：", img_path + "第" + str(i) + "张人脸")
                shape = predictor(image, d)
                feature128 = face_rec_model.compute_face_descriptor(image, shape)  # 128维特征向量
                writer.writerow(feature128)

                # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
                face = cv2.resize(face, (width, heigth))
                face_save_path = output + "/" + "face" + "/" + "pic_" + str(j) + "-" + "face_" + str(i + 1) + ".jpg"
                cv2.imwrite(face_save_path, face)
                # cv2.imshow('image', image)
                # cv2.waitKey(10)
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print("e_distance: ", dist)

    if dist > 0.4:
        return "diff"
    else:
        return "same"
def predict(img,path_csv):
    predictor, face_rec_model, detector = get_feature()
    csv_rd = pd.read_csv(path_csv, header=None)
    # 用来存放所有录入人脸特征的数组
    features_known_arr = []
    # known faces
    for i in range(csv_rd.shape[0]):
        features_someone_arr = []
        for j in range(0, len(csv_rd.ix[i, :])):
            features_someone_arr.append(csv_rd.ix[i, :][j])
        features_known_arr.append(features_someone_arr)
    print("Faces in Database：", len(features_known_arr))

    img = cv2.imread(img,1)
    # 人脸数 dets
    faces = detector(img, 1)



    # 存储所有人脸的名字
    pos_namelist = []
    name_namelist = []
    tmp = []
    # 检测到人脸
    if len(faces) != 0:
        # 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
        features_cap_arr = []
        for i in range(len(faces)):
            shape = predictor(img, faces[i])
            features_cap_arr.append(face_rec_model.compute_face_descriptor(img, shape))

        # 遍历捕获到的图像中所有的人脸



        for k in range(len(faces)):
            # 让人名跟随在矩形框的下方
            # 确定人名的位置坐标
            # 先默认所有人不认识，是 unknown
            name_namelist.append("unknown")
            # 每个捕获人脸的名字坐标
            pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

            # 对于某张人脸，遍历所有存储的人脸特征
            one = time.time()
            for i in range(len(features_known_arr)):
                print("with face_", str(i + 1), "the ", end='')
                # 将某张人脸与存储的所有人脸数据进行比对
                compare = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                if compare == "same":  # 找到了相似脸
                    name_namelist[k] = "face_" + str(i + 1)
                    tmp.append(i)
            # 矩形
            print("time is :", time.time() - one)
            print("ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")

            for kk, d in enumerate(faces):
                # 绘制矩形框
                cv2.rectangle(img, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)

        print("Name list now:", name_namelist, "\n")
        file = os.listdir("output/face")
        file.sort(key=lambda x: int(x[4:-11]))
        file_name = []
        for i in tmp:
            file_name.append(file[i])
        print("the pic is similar with:")
        print(file_name)
        # 窗口显示
        cv2.imshow("camera", img)
        cv2.waitKey(0)
    else:
        print("没有检测到人脸！")

if __name__ == '__main__':
    #以上代码封装好了，不用改，下面两个函数改参数就行，
    # get_face_feature("data2/test", "output") #用来提取人脸和特征，其中第一个参数为图片文件夹，第二个参数为输出目录，直接到一级目录就行，
    predict("0001.jpg","output/csv/feature.csv") #第一个参数为预测图片名字。第二个参数为刚刚保存的特征csv路径
#





