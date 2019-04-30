汽车检测项目
实现一：

1.本模型基于第八周作业，数据使用CompCars，来源https://blog.csdn.net/Mr_Curry/article/details/53160914?locationNum=4&fps=1，
数据集为196种车型的整体,局部外观图片，其中包括8144条训练数据和8041条测试数据，但是标注文件是matlab格式，所以需要先转换成txt文件，所以需要安装matlab
(安装方法参考https://blog.csdn.net/minione_2016/article/details/53313271),安装完成后　cd /media/taylor/E/matlab/bin  执行　./matlab


原始数据图片目录：/home/taylor/Documents/homework/vehicle-detect-dataset
标注文件目录：/home/taylor/Documents/homework/vehicle-detect-dataset/devkit

2.有了标注信息和原始图片之后，通过create_pet_tf_record.py生成tfrecord文件
因为测试数据没有类别信息，所以从训练数据中取出80%作为验证集，生成训练数据目录：/home/taylor/Documents/homework/vehicle-detect-dataset/out

3.检测框架使用faster-rcnn，基础网络模型使用inception_v2，相应的预训练模型为faster_rcnn_inception_v2_coco

4.使用1080ti，训练大约30小时

5.为验证展示制作页面：server.py,classify_image.py

6.修改visualization_utils.py,当没有车辆时给予适当的提示

7.使用方法：
cd research/

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

#python ./object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /media/taylor/G/002---study/rnn_log/output/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix /media/taylor/G/002---study/rnn_log/output/train/model.ckpt-19000  --output_directory=/media/taylor/G/002---study/rnn_log/output/exported_graphs
#python ./object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /media/taylor/G/002---study/rnn_log/output/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix /media/taylor/G/002---study/rnn_log/output/train/model.ckpt-115  --output_directory=/media/taylor/G/002---study/rnn_log/output/exported_graphs

#python ./object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /media/taylor/G/002---study/rnn_log/output/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix /media/taylor/G/002---study/ckptbak/inception_v2/model.ckpt-4100  --output_directory=/media/taylor/G/002---study/rnn_log/output/exported_graphs
python ./object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /media/taylor/G/002---study/rnn_log/output/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix /media/taylor/G/002---study/ckptbak/inception_v2/model.ckpt-14800  --output_directory=/media/taylor/G/002---study/rnn_log/output/exported_graphs
python ./object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /media/taylor/G/002---study/rnn_log/output/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix /media/taylor/G/002---study/ckptbak/inception_v2/model.ckpt-19000  --output_directory=/media/taylor/G/002---study/rnn_log/output/exported_graphs

cd object_detection/

sh ../server.sh

在浏览器中输入地址:http://0.0.0.0:5001/

预测图片保存目录：/home/taylor/Documents/homework