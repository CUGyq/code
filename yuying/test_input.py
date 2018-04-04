import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
def read_and_decode(filename,batch_size):
    files = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'channels': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })  # 取出包含image和label的feature对象
    image, label = features['img_raw'], features['label']
    height, width = features['height'], features['width']
    channels = features['channels']

    decoded_image = tf.decode_raw(image, tf.uint8)
    decoded_image = tf.reshape(decoded_image, [128,128,3])
    min_after_dequeue = 100
    capacity = 1000 + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([decoded_image, label], batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
    # image_batch = tf.cast(image_batch, tf.float32)
    return image_batch,label_batch

if __name__ == '__main__':
    image_batch,label_batch = read_and_decode("train.tfrecords",batch_size=50)
    with tf.Session() as sess:  # 开始一个会话
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1):
            a, b = sess.run([image_batch, label_batch])
            print("----------------------")
            for i in range(50):
                cv2.imshow("1",a[i])
                cv2.waitKey(1)
            print("---------------------")
        coord.request_stop()
        coord.join(threads)
