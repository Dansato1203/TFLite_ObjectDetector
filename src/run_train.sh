#! /bin/bash

python3 split_train_data.py

cd /src/models/research/object_detection_tools/data
./change_tfrecord_filename.sh

cd /src

python3 fix_pipeline.py

python3 /src/models/research/object_detection/model_main_tf2.py \
	--model_dir=/src/train_logs \
	--pipeline_config_path=/src/models/research/object_detection_tools/config/efficientdet_d0_coco17_tpu-32.config \
	--num_train_steps=100 \
	--alsologtostderr

python3 /src/models/research/object_detection/exporter_main_v2.py \
	--input_type=image_tensor \
	--pipeline_config_path=/src/models/research/object_detection_tools/config/efficientdet_d0_coco17_tpu-32.config \
	--output_directory=/src/exported_graphs \
	--trained_checkpoint_dir=/src/train_logs


#python3 /src/models/research/object_detection/export_tflite_ssd_graph.py \
#	--pipeline_config_path=/src/models/research/object_detection_tools/config/efficientdet_d0_coco17_tpu-32.config \
#  --trained_checkpoint_prefix=/content/train/model.ckpt-100 \
#  --output_directory=/src \
#  --add_postprocessing_op=true

	python3 convert_tflite.py
