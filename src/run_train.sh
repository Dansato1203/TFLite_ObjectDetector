#! /bin/bash

python3 split_train_data.py

python3 fix_pipeline.py

python3 /src/models/research/object_detection/model_main.py \
	--logtostderr=true \
	--pipeline_config_path="/src/models/research/object_detection/samples/configs/ssdlite_mobiledet_edgetpu_320x320_coco_sync_4x4.config" \
	--model_dir="/src/train_logs" 

python3 /src/models/research/object_detection/export_inference_graph.py \
	--input_type=image_tensor \
	--pipeline_config_path=/src/models/research/object_detection/samples/configs/ssdlite_mobiledet_edgetpu_320x320_coco_sync_4x4.config \
	--output_directory=/src/train_logs/inference_graph \
	--trained_checkpoint_prefix=/src/train_logs/model.ckpt-1000

python3 /src/models/research/object_detection/export_tflite_ssd_graph.py \
	--pipeline_config_path=/src/models/research/object_detection/samples/configs/ssdlite_mobiledet_edgetpu_320x320_coco_sync_4x4.config \
	--trained_checkpoint_prefix=/src/train_logs/model.ckpt-1000 \
	--output_directory=/src/train_logs/tflite \
	--add_postprocessing_op=true
