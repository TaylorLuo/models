import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

# from object_detection.utils import dataset_util
# from object_detection.utils import label_map_util

import codecs

# coding: UTF-8

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

# f1 = codecs.open('./imagessss.txt', "wb", encoding = 'utf-8')
# f2 = open('./cars_train_annos_p1_01.txt')
f2 = open('./cars_train_annos_p1_02.txt')
# f2 = open('./cars_train_annos_p1_03.txt')
# f2 = open('./cars_train_annos_p1_04.txt')

# f2 = open('./cars_validation_annos_p1_01.txt')
# f2 = open('./cars_validation_annos_p1_02.txt')
# f2 = open('./cars_validation_annos_p1_03.txt')
# f2 = open('./cars_validation_annos_p1_04.txt')
for line in f2:
    line = line[:-1].split('\t')
    im = line[-1].split('.')[0]
    l = line[-2]
    bbox = line[0:4]
    for idx, value in enumerate(bbox):
      bbox[idx] = float(bbox[idx])
    # f1.writelines(('data/obj/'+im +'\n'))
    # print(line)
    b = (float(bbox[1]*432), float(bbox[3]*432), float(bbox[0]*320),float(bbox[2]*320))
    bb = convert((432, 320), b)
    # out_file = open('/media/taylor/G/002---study/rnn_log/output/train_pics/train_pics_yolo9000_p1_01/%s.txt'%(im), 'w')
    out_file = open('/media/taylor/G/002---study/rnn_log/output/train_pics/train_pics_yolo9000_p1_02/%s.txt'%(im), 'w')
    # out_file = open('/media/taylor/G/002---study/rnn_log/output/train_pics/train_pics_yolo9000_p1_03/%s.txt'%(im), 'w')
    # out_file = open('/media/taylor/G/002---study/rnn_log/output/train_pics/train_pics_yolo9000_p1_04/%s.txt'%(im), 'w')

    # out_file = open('/media/taylor/G/002---study/rnn_log/output/validation_pics/validation_pics_yolo9000_p1_01/%s.txt'%(im), 'w')
    # out_file = open('/media/taylor/G/002---study/rnn_log/output/validation_pics/validation_pics_yolo9000_p1_02/%s.txt'%(im), 'w')
    # out_file = open('/media/taylor/G/002---study/rnn_log/output/validation_pics/validation_pics_yolo9000_p1_03/%s.txt'%(im), 'w')
    # out_file = open('/media/taylor/G/002---study/rnn_log/output/validation_pics/validation_pics_yolo9000_p1_04/%s.txt'%(im), 'w')
    out_file.write(str(l) + " " + " ".join([str(a) for a in bb]) + '\n')

out_file.close()
f2.close()

#0.3535040020942688	0.08965688943862915	0.7543930411338806	0.9647945761680603	582	18_Label_582.jpg
# xmin = float(bboxs[1])
# xmax = float(bboxs[3])
# ymin = float(bboxs[0])
# ymax = float(bboxs[2])
# yolo require order

