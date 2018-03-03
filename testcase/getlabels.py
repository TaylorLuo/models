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


def read_examples_list(path):
  """Read list of training or validation examples.

  The file is assumed to contain a single example per line where the first
  token in the line is an identifier that allows us to find the image and
  annotation xml for that example.

  For example, the line:
  xyz 3
  would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

  Args:
    path: absolute path to examples list file.

  Returns:
    list of example identifiers (strings).
  """
  with tf.gfile.GFile(path) as fid:
    lines = fid.readlines()
  return [line.strip().split('	')[0] for line in lines]

data_dir = './'
annotations_dir = os.path.join(data_dir, 'devkit')
examples_path = os.path.join(annotations_dir, 'labels_items_pet.txt')
examples_list = read_examples_list('./labels_items_pet.txt')
f = open('./labels_items3.txt', 'w', encoding='utf8')
for idx, item in enumerate(examples_list):
    feature_dict = 'item {\n  id: ' + str(idx + 1) + '\n  '+ 'name: ' +"'" +str(item) + "'" + '\n}\n'
    f.writelines(feature_dict + '\n')
f.close()

# label_map_dict = label_map_util.get_label_map_dict('./labels_items2.txt')