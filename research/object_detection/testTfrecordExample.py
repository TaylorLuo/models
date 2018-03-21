import tensorflow as tf
from google.protobuf.json_format import MessageToJson
from object_detection.metrics import tf_example_parser
import classify_image as cls

file = "/media/taylor/H/CSDN/project_datasets/project3_vehicle_detection/pj_vehicle_train_00000-of-00004.tfrecord"
record_iterator = tf.python_io.tf_record_iterator(path=file)
data_parser = tf_example_parser.TfExampleDetectionAndGTParser()
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    decoded_dict = data_parser.parse(example)
    print("over")