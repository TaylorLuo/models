import tensorflow as tf
from google.protobuf.json_format import MessageToJson
from object_detection.metrics import tf_example_parser

file = "/media/taylor/H/CSDN/project_datasets/project3_vehicle_detection/pj_vehicle_train_00000-of-00004.tfrecord"
fileNum=1
for example in tf.python_io.tf_record_iterator(file):
    jsonMessage = MessageToJson(tf.train.Example.FromString(example))
    data_parser = tf_example_parser.TfExampleDetectionAndGTParser()
    decoded_dict = data_parser.parse(example)
    with open("/media/taylor/G/002---study/rnn_log/output/out-images/image_{}".format(fileNum),"w") as text_file:
        print(jsonMessage,file=text_file)
    fileNum+=1