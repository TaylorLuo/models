# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from PIL import Image
from matplotlib import pyplot as plt
import time

NUM_CLASSES = 196

# def parse_args(check=True):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--output_dir', type=str, required=True)
#     parser.add_argument('--dataset_dir', type=str, required=True)
#     FLAGS, unparsed = parser.parse_known_args()
#     return FLAGS, unparsed

FLAGS = tf.app.flags.FLAGS

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self, label_path=None):
    if not label_path:
      tf.logging.fatal('please specify the label file.')
      return
    self.node_lookup = self.load(label_path)

  def load(self, label_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(label_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(label_path).readlines()
    id_to_human = {}
    for line in proto_as_ascii_lines:
      if line.find(':') < 0:
        continue
      _id, human = line.rstrip('\n').split(':')
      id_to_human[int(_id)] = human

    return id_to_human

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph(model_file=None):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  if not model_file:
    model_file = FLAGS.model_file
  with open(model_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def test(name):
    print(name)

def run_inference_on_image(image, model_file=None):
    # FLAGS, unparsed = parse_args()
    PATH_TO_CKPT = os.path.join(FLAGS.model_dir, 'exported_graphs/frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'devkit/labels_items.txt')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # test_img_path = os.path.join(FLAGS.dataset_dir, 'test.jpg')
    # print(test_img_path)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = Image.open(image)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # print('11############################')
            # print(classes)
            # print(scores)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            image_outpath = os.path.join(FLAGS.dataset_dir, 'output-'+str(int(time.time()))+'.png')
            plt.imsave(image_outpath, image_np)

    agnostic_mode = False
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)
    lookslike = ''
    for i in range(min(1, boxes.shape[0])):
        if scores is None or scores[i] > 0.5:
            if scores is None:
                print('score is none')
            else:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%'.format(
                        class_name,
                        int(100 * scores[i]))
                else:
                    display_str = 'score: {}%'.format(int(100 * scores[i]))
        else:
            if scores[i] == max(scores):
                box = tuple(boxes[i].tolist())
                if scores is None:
                    print('score is none')
                else:
                    if not agnostic_mode:
                        class_name = 'NoCar'
                        display_str = '{}: {}%'.format(
                            class_name,
                            int(100 * scores[i]))
                    else:
                        display_str = 'score: {}%'.format(int(100 * scores[i]))

            lookslike = ('But it is like class: %s, info: %s' % (
             classes[i], category_index[classes[i]]['name']))

    return max(scores), class_name, image_outpath, lookslike


def main(_):
  print('@@@@@@@@@@@@@@@@@@@@@@@@@')
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  run_inference_on_image(image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_file',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to the .pb file that contains the frozen weights. \
      """
  )
  parser.add_argument(
      '--label_file',
      type=str,
      default='',
      help='Absolute path to label file.'
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
