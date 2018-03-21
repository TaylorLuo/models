import tensorflow as tf
from google.protobuf.json_format import MessageToJson
from object_detection.metrics import tf_example_parser
from object_detection import classify_image as cls
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import PIL.Image
import os.path
from matplotlib import pyplot as plt

file = "/media/taylor/H/CSDN/project_datasets/project3_vehicle_detection/pj_vehicle_train_00000-of-00004.tfrecord"
output_path = "/media/taylor/G/002---study/rnn_log/output/out-images"
record_iterator = tf.python_io.tf_record_iterator(path=file)
data_parser = tf_example_parser.TfExampleDetectionAndGTParser()
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    decoded_dict = data_parser.parse(example)
    print("over")



def run_inference_on_image(label_num, encoded_jpg_io, model_file=None):
    PATH_TO_LABELS = "/media/taylor/H/CSDN/project_datasets/project3_vehicle_detection/labels.txt"
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=764,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # test_img_path = os.path.join(FLAGS.dataset_dir, 'test.jpg')
    # print(test_img_path)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = PIL.Image.open(encoded_jpg_io)
            image_np = np.asarray(image)
            # image = Image.open(image)
            # image_np = load_image_into_numpy_array(image)
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
            image_outpath = os.path.join(output_path, 'output-'+str(label_num)+'.png')
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