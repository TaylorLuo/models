import tensorflow as tf
import codecs

def read_examples_list(path):
  with tf.gfile.GFile(path) as fid:
    lines = fid.readlines()
  return [line.strip().split('	')[0] for line in lines]

def get_label_map_dict(path):
    examples_list = read_examples_list(path)
    label_map_dict = {}
    for idx, item in enumerate(examples_list):
        valuestr = str(item.split(':')[1])
        label_map_dict[idx] = valuestr
    return label_map_dict

print(get_label_map_dict('./labels.txt')[2])