import cv2
import dlib
import os
import numpy as np
import csv
import pandas as pd
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
            image = cv2.imread(img_path, 1)
            dets = detector(image, 1)
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = image[x1:y1, x2:y2]
                cv2.rectangle(image, (x2, x1), (y2, y1), (0, 255, 0), 1)
                print("正在读的人脸图像：", img_path + "第" + str(i) + "张人脸")
                shape = predictor(image, d)
                feature128 = face_rec_model.compute_face_descriptor(image, shape)  # 128维特征向量
                writer.writerow(feature128)

                # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
                face = cv2.resize(face, (width, heigth))
                face_save_path = output + "/" + "face" + "/" + "person_" + str(j) + "-" + "face_" + str(i + 1) + ".jpg"
                cv2.imwrite(face_save_path, face)
                cv2.imshow('image', image)
                cv2.waitKey(10)


    # img = cv2.imread("1.jpg",1)
    # dets = detector(img, 1)
    # for i, d in enumerate(dets):
    #     x1 = d.top() if d.top() > 0 else 0
    #     y1 = d.bottom() if d.bottom() > 0 else 0
    #     x2 = d.left() if d.left() > 0 else 0
    #     y2 = d.right() if d.right() > 0 else 0
    #
    #     face = img[x1:y1, x2:y2]
    #     cv2.rectangle(img, (x2, x1), (y2, y1), (0, 255, 0), 1)
    #     shape = predictor(img, d)
    #     feature128 = face_rec_model.compute_face_descriptor(img, shape)  # 128维特征向量
    #     print(feature128)
    #     # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
    #     face = cv2.resize(face, (width, heigth))
    #     cv2.imshow('image', img)
    #     cv2.waitKey(0)
def get_features_into_CSV(input,output,width = 64,heigth = 64):
    if not os.path.exists(input):
        print("输入路径错误！")
    if not os.path.exists(output):
        os.mkdir(output)
    predictor, face_rec_model, detector = get_feature()
    for person in os.listdir(input):
        person_csv = output+"/"+person+".csv"
        dir_pics = os.listdir(os.path.join(input,person))  # 得到每一个人的图片
        with open(person_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(dir_pics)):
            # 调用return_128d_features()得到128d特征
                print("正在读的人脸图像：", os.path.join(os.path.join(input,person),dir_pics[i]))
                features_128d = return_128d_features(os.path.join(os.path.join(input,person),dir_pics[i]),detector,predictor,face_rec_model)
                # 遇到没有检测出人脸的图片跳过
                if features_128d == 0:
                    i += 1
                else:
                    writer.writerow(features_128d)
# 返回单张图像的 128D 特征
def return_128d_features(path_img,detector,predictor,facerec):
    img = cv2.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)
    print("检测的人脸图像：", path_img, "\n")
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")
    return face_descriptor
def  get_mean(input,output):
    if not os.path.exists(input):
        print("输入路径错误！")
    with open(output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        csv_rd = os.listdir(input)
        print("得到的特征均值 / The generated average values of features stored in: ")
        for i in range(len(csv_rd)):
            path_csv = os.path.join(input,csv_rd[i])
            column_names = []
            # 128列特征
            for feature_num in range(128):
                column_names.append("features_" + str(feature_num + 1))
            # 利用pandas读取csv
            rd = pd.read_csv(path_csv, names=column_names)
            # 存放128维特征的均值
            feature_mean = []
            for feature_num in range(128):
                tmp_arr = rd["features_" + str(feature_num + 1)]
                tmp_arr = np.array(tmp_arr)
                # 计算某一个特征的均值
                tmp_mean = np.mean(tmp_arr)
                feature_mean.append(tmp_mean)
            writer.writerow(feature_mean)
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print("e_distance: ", dist)

    if dist > 0.4:
        return "diff"
    else:
        return "same"
# 返回一张图像多张人脸的 128D 特征
def get_128d_features(img_gray,predictor, face_rec_model, detector):
    dets = detector(img_gray, 1)
    if len(dets) != 0:
        face_des = []
        for i in range(len(dets)):
            shape = predictor(img_gray, dets[i])
            face_des.append(face_rec_model.compute_face_descriptor(img_gray, shape))
    else:
        face_des = []
    return face_des
def predict(img,path_mean_csv):
    predictor, face_rec_model, detector = get_feature()
    csv_rd = pd.read_csv(path_mean_csv, header=None)
    # 用来存放所有录入人脸特征的数组
    features_known_arr = []
    # known faces
    for i in range(csv_rd.shape[0]):
        features_someone_arr = []
        for j in range(0, len(csv_rd.ix[i, :])):
            features_someone_arr.append(csv_rd.ix[i, :][j])
        features_known_arr.append(features_someone_arr)
    print("Faces in Database：", len(features_known_arr))

    img = cv2.imread(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 人脸数 dets
    faces = detector(img_gray, 0)
    #用于显示而已
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, "Press 'q': Quit", (20, 400), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
    # 存储所有人脸的名字
    pos_namelist = []
    name_namelist = []
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
            for i in range(len(features_known_arr)):
                print("with person_", str(i + 1), "the ", end='')
                # 将某张人脸与存储的所有人脸数据进行比对
                compare = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                if compare == "same":  # 找到了相似脸
                    name_namelist[k] = "person_" + str(i + 1)

            # 矩形框
            for kk, d in enumerate(faces):
                # 绘制矩形框
                cv2.rectangle(img, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)

        # 写人脸名字
        for i in range(len(faces)):
            cv2.putText(img, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

    print("Name list now:", name_namelist, "\n")

    cv2.putText(img, "Face Register", (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    # 按下 q 键退出
    if kk == ord('q'):
        cv2.destroyAllWindows()
    # 窗口显示
    cv2.imshow("camera", img)
    cv2.waitKey(0)







if __name__ == '__main__':
    # get_features_into_CSV("data", "output/person", width=64, heigth=64) #求人的人脸特征，保存大csv
    # get_mean("output/person","output/mean/s.csv")#计算每一个人的人脸的均值，得到匹配模板库
    predict("2.jpg","output/mean/features_all.csv")#进行预测





