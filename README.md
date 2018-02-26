本地运行方法：

前提环境：python3.5 Tensorflow1.4

1.登录自己的github fork models项目，然后用pycharmcheckout到本地
2.下载数据集quiz-w8-doc，并将inference.py、run.py、run.sh、ssd_mobilenet_v1_pets.config添加到models/research目录下
3.参照https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md安装Object Detection API所需要的libs
并在research目录下执行脚本：protoc object_detection/protos/*.proto --python_out=. 编译Protobuf，用脚本：python object_detection/builders/model_builder_test.py
验证是否编译成功
4.修改create_pet_tf_record.py，并在research目录下执行环境变量配置脚本：export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
5.到https://gitee.com/ai100/quiz-w8-data.git下载实验数据
6.执行生成tfrecord脚本：python object_detection/dataset_tools/create_pet_tf_record.py --label_map_path=/home/taylor/Documents/homework/week08/quiz-w8-data/labels_items.txt --data_dir=/home/taylor/Documents/homework/week08/quiz-w8-data --output_dir=/home/taylor/Documents/homework/week08/quiz-w8-data/out
7.修改ssd_mobilenet_v1_pets.config、run.sh中目录及其他配置
8.执行 python run.py


tinymind运行方法：

1.新建数据集my-objectdetection，并将model.ckpt.data-00000-of-00001、model.ckpt.index、model.ckpt.meta、labels_items.txt、pet_val.record 、pet_train.record
test.jpg、ssd_mobilenet_v1_pets.config上传
2.修改run.sh、ssd_mobilenet_v1_pets.config中的目录配置:output_dir dataset_dir和label_map_path input_path fine_tune_checkpoint
3.新建模型
4.运行

tinymind模型地址：https://www.tinymind.com/luoweile/myobjectdetection
