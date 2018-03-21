# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 23:53:56 2017

@author: zhangxu
"""
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from object_detection.utils import visualization_utils as vis_util

#写入图片路径
swd = '/media/taylor/G/002---study/rnn_log/output/out-images/'
#TFRecord文件路径
data_path = '/media/taylor/H/CSDN/project_datasets/project3_vehicle_detection/pj_vehicle_train_00000-of-00004.tfrecord'
# 获取文件名列表
data_files = tf.gfile.Glob(data_path)
print(data_files)
# 文件名列表生成器

filename_queue = tf.train.string_input_producer(data_files,shuffle=True)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'image/class/label': tf.FixedLenFeature([], tf.int64),
                                       'image/encoded' : tf.FixedLenFeature([], tf.string),
                                       'image/width': tf.FixedLenFeature([], tf.int64),
                                       'image/height': tf.FixedLenFeature([], tf.int64),
                                   })  #取出包含image和label的feature对象
#tf.decode_raw可以将字符串解析成图像对应的像素数组
image = tf.image.decode_jpeg(features['image/encoded'])
height = tf.cast(features['image/height'],tf.int32)
width = tf.cast(features['image/width'],tf.int32)
label = tf.cast(features['image/class/label'], tf.int32)
channel = 3
# image = tf.reshape(image, [height,width,channel])


with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    #启动多线程
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)

    # PATH_TO_CKPT = os.path.join("/media/taylor/G/002---study/rnn_log/output", 'exported_graphs_inception04/frozen_inference_graph.pb')
    PATH_TO_CKPT = os.path.join("/home/taylor/Documents/homework/week08", 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb')

    for i in range(1500):
        #image_down = np.asarray(image_down.eval(), dtype='uint8')
        # plt.imshow(image.eval())
        # plt.show()
        images,heights, widths, labels = sess.run([image,height,width,label])#在会话中取出image和label
        print("2222@@@@@@@@@@@@")
        print("第====%d====张图片", i)
        # img=Image.fromarray(single, 'RGB')#这里Image是之前提到的
        # img.save(swd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
        print("图片高-宽-标签")
        print(heights,widths, labels)
        # plt.imsave((swd+str(i)+'_''Label_'+str(labels)+'.jpg'), images)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess2:
                init_op2 = tf.initialize_all_variables()
                sess2.run(init_op2)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                image_np_expanded = np.expand_dims(images, axis=0)
                (boxes, scores, classes, num) = sess2.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                # print('11############################')
                # print(boxes)
                for i in range(min(1, boxes.shape[0])):
                    if scores is None or scores[i] > 0.1:
                        if scores[i] == max(scores):
                            print("3333@@@@@@@@@@@@")
                            print(max(scores))
                            box = tuple(boxes[i].tolist())
                            # print("4444@@@@@@@@@@@@")
                            # print(box)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    images,
                    boxes,
                    [],
                    scores,
                    {},
                    use_normalized_coordinates=True,
                    line_thickness=8)
                plt.imsave((swd + str(i) + '_''Label_' + str(labels) + '.png'), images)

    coord.request_stop()
    coord.join(threads)