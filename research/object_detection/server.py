# coding=utf-8
import os
import sys
import importlib
importlib.reload(sys)
import time
from flask import request, send_from_directory
from flask import Flask, request, redirect, url_for
import uuid
import tensorflow as tf
import classify_image as cls

ALLOWED_EXTENSIONS = set(['jpg','JPG', 'jpeg', 'JPEG', 'png'])

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '/media/taylor/G/002---study/rnn_log/output', """Path to graph_def pb, """)
tf.app.flags.DEFINE_string('dataset_dir', '/home/taylor/Documents/homework/vehicle-detect-dataset', """Path to graph_def pb, """)
tf.app.flags.DEFINE_string('model_name', 'frozen_inference_graph.pb', '')
tf.app.flags.DEFINE_string('label_file', 'labels_items4.txt', '')
tf.app.flags.DEFINE_string('upload_folder', '/home/taylor/Documents/homework/vehicle-detect-dataset', '')
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")
tf.app.flags.DEFINE_integer('port', '5001',
        'server with port,if no port, use deault port 80')

tf.app.flags.DEFINE_boolean('debug', False, '')

UPLOAD_FOLDER = FLAGS.upload_folder
ALLOWED_EXTENSIONS = set(['jpg','JPG', 'jpeg', 'JPEG', 'png'])

app = Flask(__name__)
app._static_folder = UPLOAD_FOLDER

def allowed_files(filename):
  return '.' in filename and \
      filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def rename_filename(old_file_name):
  basename = os.path.basename(old_file_name)
  name, ext = os.path.splitext(basename)
  new_name = str(uuid.uuid1()) + ext
  return new_name

def inference(file_name):
  try:
    score, top_names, image_outpath, lookslike = cls.run_inference_on_image(file_name, model_file=FLAGS.model_name)
    print('@@@@@@@@@@@@@@@@@@@@@@')
    print(score)
    print(top_names)
  except Exception as ex:
    print(ex)
    return ""
  new_url = '/static/%s' % os.path.basename(file_name)
  print('##################')
  print(file_name)
  print(image_outpath)
  image_tag = '<img src="%s"></img>'
  new_tag = image_tag % new_url
  new_url2 = '/static/%s' % os.path.basename(image_outpath)
  image_tag2 = '<img src="%s"></img><p>'
  new_tag2 = image_tag2 % new_url2
  format_string = ''
  format_string += '%s (score:%.5f)<BR>' % (top_names, score)
  ret_string = new_tag  + new_tag2 + format_string + lookslike +'<BR>'
  return ret_string


@app.route("/", methods=['GET', 'POST'])
def root():
  result = """
    <!doctype html>
    <title>My Vehicle Detection Demo</title>
    <h1 style='color:red;bold'>Inception_V2+Faster-RCNN+CompCars</h1>
    <h1>请导入汽车图片</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file value='选择图片'>
         <input type=submit value='上传检测'>
    </form>
    <p>%s</p>
    """ % "<br>"
  if request.method == 'POST':
    file = request.files['file']
    old_file_name = file.filename
    if file and allowed_files(old_file_name):
      filename = rename_filename(old_file_name)
      file_path = os.path.join(UPLOAD_FOLDER, filename)
      file.save(file_path)
      type_name = 'N/A'
      print('file saved to %s' % file_path)
      out_html = inference(file_path)
      return result + out_html
  return result

if __name__ == "__main__":
  print('listening on port %d' % FLAGS.port)
  app.run(host='0.0.0.0', port=FLAGS.port, debug=FLAGS.debug, threaded=True)

