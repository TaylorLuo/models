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


examples_list = read_examples_list('./cars_train_annos_p1_01.txt')
f1 = codecs.open('./train.txt', "wb", encoding = 'utf-8')
f2 = open('./cars_train_annos_p1_01.txt')
for line in f2:
    line = line[:-1].split('\t')
    im = line[-1]
    f1.writelines(('data/obj/'+im +'\n'))
f1.close()
f2.close()



