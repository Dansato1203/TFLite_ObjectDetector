#! /bin/bash

python3 split_train_data.py

#./change_tfrecord_filename.sh

./make_numberlist.sh

python3 /src/voc2012.py \
	--data_dir '/src/dataset' \
	--split train \
	--output_file /src/dataset/dataset_train.tfrecord \
	--classes /src/dataset/label.names \

python3 /src/voc2012.py \
	--data_dir '/src/dataset' \
	--split val \
	--output_file /src/dataset/dataset_val.tfrecord \
	--classes /src/dataset/label.names \

python3 /src/yolov3-tf2/convert.py \
	--weights /src/pretrained_model/yolov3.weight

python3 /src/yolov3-tf2/train.py \
	--dataset /src/dataset/dataset_train.tfrecord \
	--val_dataset /src/dataset/dataset_val.tfrecord \
	--classes /src/dataset/label.names \
	--num_classes 3 \
	--mode fit --transfer darknet \
	--batch_size 16 \
	--epochs 20 \
	--weights /src/checkpoints/yolov3.tf \
	--weights_num_classes 80 

python3 /src/yolov3-tf2/export_tflite.py \
	--weights /src/checkpoints/yolov3_train_4.tf \
	--output /src/checkpoints/yolov3.tflite \
	--classes /src/dataset/label.names \
	--image /src/0053.jpg \
	--num_classes 3

edgetpu_compiler /src/checkpoints/yolov3.tflite
