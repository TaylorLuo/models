import chardet

#!/usr/bin/python
# coding: utf-8


import logging

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

def load_labelmap(path):
  """Loads label map proto.

  Args:
    path: path to StringIntLabelMap proto text file.
  Returns:
    a StringIntLabelMapProto
  """
  with tf.gfile.GFile(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
      text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
      label_map.ParseFromString(label_map_string)
  # _validate_label_map(label_map)
  return label_map


label_map = load_labelmap("./labels_items3.txt")