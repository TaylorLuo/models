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

def read_examples_list(path):
  """Read list of training or validation examples.

  """
  with tf.gfile.GFile(path) as fid:
    lines = fid.readlines()
  return [line.strip().split('	')[0] for line in lines]


examples_list = read_examples_list('./labels.txt')


#### get9ktree ####
f = codecs.open('./9ktree4.txt', "wb", encoding = 'utf-8')
for idx, item in enumerate(examples_list):
    strss = item.split(':')[1]
    f.writelines((strss + ' 3802' +'\n'))
    # f.write(('"'+strss + '",'))
f.close()

#### get9klabels ####
# f = codecs.open('./9labels.txt', "wb", encoding = 'utf-8')
# for idx, item in enumerate(examples_list):
#     strss = item.split(':')[0]
#     f.writelines((strss + '\n'))
# f.close()


# examples_list = read_examples_list('./yolo9000/9k.tree')
# f = codecs.open('./9klabels2.txt', "wb", encoding = 'utf-8')
# for idx, item in enumerate(examples_list):
#     strss = item.split(':')[1]
#     f.writelines((strss + ' 3802' +'\n'))
#     # f.writelines((strss +'\n'))
# f.close()




