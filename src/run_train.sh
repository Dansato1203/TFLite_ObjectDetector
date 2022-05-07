#! /bin/bash

python3 split_train_data.py

python3 fix_pipeline.py

python3 /src/models/research/object_detection/model_main.py \
	--pipeline_config_path="/src/models/research/object_detection/samples/configs/ssdlite_mobiledet_edgetpu_320x320_coco_sync_4x4.config" \
	#--model_dir="/src/train_result" \
	--model_dir=/src/train

python3 /content/models/research/object_detection/export_inference_graph.py \
	--input_type=image_tensor \
	--pipeline_config_path=/src/models/research/object_detection/samples/configs/ssdlite_mobiledet_edgetpu_320x320_coco_sync_4x4.config \
	--output_directory=/src/inference_graph \
	--trained_checkpoint_prefix=/content/train/model.ckpt-10000
