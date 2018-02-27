# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

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

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '/home/taylor/Documents/homework/vehicle-detect-dataset/devkit/labels_items2.txt.txt','Path to label map proto')

FLAGS = flags.FLAGS

def read_imagefile_label(imagefile_label):
  """Reads .txt files and returns lists of imagefiles paths and labels
  Args:
    imagefile_label: .txt file with image file paths and labels in each line
  Returns:
    imagefile: list with image file paths
    label: list with labels
  """
  f = open(imagefile_label)
  imagefiles = []
  labels = []
  bboxs = []
  for line in f:
    line = line[:-1].split(' ')
    im = line[0]
    l = line[1]
    bbox = line[2:]
    for idx, value in enumerate(bbox):
      bbox[idx] = int(bbox[idx])
    imagefiles.append(im)
    labels.append(int(l))
    bboxs.append(bbox)
  return {'filename': imagefiles, 'class': labels, 'bboxs': bboxs}

def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).


  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """


  img_path = os.path.join(image_subdirectory, data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(image.shape[1])
  height = int(image.shape[0])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []

  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []

  bboxs = data['bboxs']
  xmin = int(bboxs[0])
  xmax = int(bboxs[1])
  ymin = int(bboxs[2])
  ymax = int(bboxs[3])

  xmins.append(xmin / width)
  ymins.append(ymin / height)
  xmaxs.append(xmax / width)
  ymaxs.append(ymax / height)

  class_name = data['class']
  classes_text.append(class_name.encode('utf8'))
  classes.append(label_map_dict[class_name])
  truncated.append(0)
  poses.append('Unspecified'.encode('utf8'))

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses)
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     image_dir,
                     data):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """

  writer = tf.python_io.TFRecordWriter(output_filename)
  try:
    tf_example = dict_to_tf_example(
        data, label_map_dict, image_dir)
    writer.write(tf_example.SerializeToString())
  except ValueError:
    logging.warning('Invalid example: %s, ignoring.', "with Error when writing tfrecord")

  writer.close()


# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir #/home/taylor/Documents/homework/vehicle-detect-dataset
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from Pet dataset.')
  image_dir_train = os.path.join(data_dir, 'cars_train')
  image_dir_test = os.path.join(data_dir, 'cars_test')
  annotations_dir = os.path.join(data_dir, 'devkit')
  examples_path_train = os.path.join(annotations_dir, 'cars_train_annos.txt')
  examples_path_test = os.path.join(annotations_dir, 'cars_test_annos.txt')
  # examples_list_train = dataset_util.read_examples_list(examples_path_train)
  # examples_list_test = dataset_util.read_examples_list(examples_path_test)
  # print(len(examples_list_train))
  # print(len(examples_list_test))

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  # random.seed(42)
  # random.shuffle(examples_list_train)
  # random.shuffle(examples_list_test)
  # train_examples = examples_list_train
  # val_examples = examples_list_test
  # logging.info('%d training and %d validation examples.',
  #              len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'pet_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'pet_val.record')

  data_train = read_imagefile_label(examples_path_train)
  data_test = read_imagefile_label(examples_path_test)

  create_tf_record(train_output_path, label_map_dict,
                   image_dir_train, data_train)
  create_tf_record(val_output_path, label_map_dict,
                   image_dir_test, data_test)


if __name__ == '__main__':
  tf.app.run()
