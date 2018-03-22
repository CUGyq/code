import cv2
import numpy as np
import os.path
import tensorflow as tf
import train
import time
def reg(img):
    label = None
    acc = 0.0
    img0 = cv2.resize(img, (64, 64))
    img2 = tf.cast(img0, tf.float32)
    img3 = tf.reshape(img2, (1, 64, 64, 3))
    logit = train.inference(img3)
    qq = tf.nn.softmax(logit)
    maxa = tf.argmax(logit, 1)
    q = qq[0][maxa[0]]
    variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
    variable_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            for i in range(1):
                ss, dd = sess.run([maxa, q])
                label = ss
                acc = dd
    return label,acc

i=52
cascade_path ="D:\opencv\sources\data\haarcascades\haarcascade_frontalcatface.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)
img = cv2.imread("face2/52.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
MOVING_AVERAGE_DECAY = 0.99
EVAL_INTERVAL_SECS = 10
# image_size = 64
# img = cv2.imread("./pre/cao/540.jpg")
# a,b = reg(img)
# print(a)
face = faceCascade.detectMultiScale(gray, 1.1, 1, 0)
if len(face) > 0:
    for rect in face:
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        pre_img = img[y:y + h, x:x + w]
        # cv2.imwrite( "aaa" + '.jpg', pre_img)  # 存储为图像
        ss,dd = reg(pre_img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if ss == 0:
            cv2.putText(img, "cao!,a = %g"%dd, (x-2, y-2), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.imwrite("./face3/" + str(i) + '.jpg', img)  # 存储为图像
            # cv2.imshow("a", img)
            # cv2.waitKey(0)
        elif ss == 1:
            cv2.putText(img, "hu!,a = %g" % dd, (x-2, y-2), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.imwrite("./face3/" + str(i) + '.jpg', img)  # 存储为图像
        #     cv2.imshow("a", img)
        #     cv2.waitKey(0)
        else:
            cv2.putText(img, "mei!,a = %g" % dd, (x-2, y-2), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, ), 2)
            cv2.imwrite("./face3/" + str(i) + '.jpg', img)  # 存储为图像
        #     cv2.imshow("a",img)
        #     cv2.waitKey(0)
else:
    print("No checkpoint file found")


