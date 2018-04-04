import tensorflow as tf
import numpy as np
import os
import train
import time
import cv2
MOVING_AVERAGE_DECAY = 0.99
EVAL_INTERVAL_SECS = 10
image_size = 64
img = cv2.imread("data/test/train29992.png")
img0 = cv2.resize(img, (64, 64))
img2 = tf.cast(img0,tf.float32)
img3 = tf.reshape(img2,(1,64,64,3))
y = train.inference(img3)
qq = tf.nn.softmax(y)
maxa = tf.argmax(y,1)
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
        saver.restore(sess,ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        for i in range(1):
            ss,dd= sess.run([maxa,q])
            cv2.putText(img, "class = %g " % ss, (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            cv2.imshow("a", img)
            cv2.waitKey(0)
            # print(ss)

            # if ss == 0:
            #     print("这是cao!,正确率是%g",dd)
            # elif ss == 1:
            #     print("这是hu!,正确率是%g",dd)
            # else:
            #     print("这是mei,正确率是%g",dd)
    else:
        print("No checkpoint file found")
    # time.sleep(EVAL_INTERVAL_SECS)
