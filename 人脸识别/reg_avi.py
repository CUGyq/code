import cv2
import numpy as np
import os.path
import tensorflow as tf
import train
import time
cascade_path ="D:\opencv\sources\data\haarcascades\haarcascade_frontalcatface.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)
MOVING_AVERAGE_DECAY = 0.99
EVAL_INTERVAL_SECS = 10
c = 0
cap = cv2.VideoCapture('cao.avi')

while (cap.isOpened()):
    c += 1
    ret, frame = cap.read()
    if (ret == 0):
        break;

    if len(frame.shape) == 3 or len(frame.shape) == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    face = faceCascade.detectMultiScale(gray, 1.1, 1)
    if len(face) > 0:
        x = face[0][0]
        y = face[0][1]
        w = face[0][2]
        h = face[0][3]

#         cv2.imwrite("./face1/" +"g"+ str(c - 1) + '.jpg',frame[y:y + h, x:x + w] )  # 存储为图像
        pre_img = frame[y:y + h, x:x + w]
        img0 = cv2.resize(pre_img, (64, 64))
#         cv2.imwrite("./face1/" + "g" + str(c - 1) + '.jpg', img0)  # 存储为图像
        with tf.Graph().as_default() as g:
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
                    ss, dd = sess.run([maxa, q])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if ss == 0:
                        cv2.putText(frame, "cao!,a=%g" % dd, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
                        cv2.imshow("a", frame)
                        cv2.waitKey(1)
                    elif ss == 1:
                        cv2.putText(frame, "hu!,a= %g" % dd, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                                    1)
                        cv2.imshow("a", frame)
                        cv2.waitKey(1)

                    else:
                        cv2.putText(frame, "mei!,a= %g" % dd, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                                    1)
                        cv2.imshow("a", frame)
                        cv2.waitKey(1)
                else:
                    print("No checkpoint file found")
    else:
        print("NO face!")
