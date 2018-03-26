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


import json


def dump(lst):
    fp = open("./labels_items5.txt", "w")
    fp.write(json.dumps(lst))
    fp.close()

examples_list = read_examples_list('./labels.txt')
f = codecs.open('./labels_items4.txt', "wb", encoding = 'utf-8')
# f2 = codecs.open('./labels_items4.txt',"wb",encoding = 'utf-8')
for idx, item in enumerate(examples_list):
    strss = item.split(':')[1]
    # udata = strss.decode("utf-8")
    asciidata = strss.encode("ascii", "ignore")
    feature_dict = 'item {\n  id: ' + str(idx + 1) + '\n  '+ 'name: ' +"'" +str(asciidata) + "'" + '\n}\n'
    f.writelines((feature_dict + '\n'))
    # dump(feature_dict + '\n')
    # f2.write(feature_dict + '\n')
f.close()



