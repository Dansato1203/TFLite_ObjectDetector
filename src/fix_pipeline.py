import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
import os

NUM_STEPS = 1000
BATCH_SIZE = 8
NUM_CLASSES = 3


pipeline = pipeline_pb2.TrainEvalPipelineConfig()
config_path = '/src/models/research/object_detection/samples/configs/ssdlite_mobiledet_edgetpu_320x320_coco_sync_4x4.config'

with tf.gfile.GFile(config_path, "r") as f:
	proto_str = f.read()
	text_format.Merge(proto_str, pipeline)

pipeline.train_input_reader.tf_record_input_reader.input_path[:] = ['/src/models/research/train_data/*.tfrecord']
pipeline.train_input_reader.label_map_path = '/src/pretrained_model/pbtxt/tf_label_map.pbtxt'
pipeline.eval_input_reader[0].tf_record_input_reader.input_path[:] = ['/src/models/research/val_data/*.tfrecord']
pipeline.eval_input_reader[0].label_map_path = '/src/pretrained_model/pbtxt/tf_label_map.pbtxt'
pipeline.train_config.fine_tune_checkpoint = '/src/pretrained_model/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19/fp32/model.ckpt'
pipeline.train_config.batch_size = BATCH_SIZE
pipeline.train_config.num_steps = NUM_STEPS
pipeline.model.ssd.num_classes = NUM_CLASSES
# Enable ssdlite, this should already be enabled in the config we downloaded, but this is just to make sure.
pipeline.model.ssd.box_predictor.convolutional_box_predictor.kernel_size = 3
pipeline.model.ssd.box_predictor.convolutional_box_predictor.use_depthwise = True
pipeline.model.ssd.feature_extractor.use_depthwise = True
# Quantization Aware Training
pipeline.graph_rewriter.quantization.delay = 0
pipeline.graph_rewriter.quantization.weight_bits = 8
pipeline.graph_rewriter.quantization.activation_bits = 8

config_text = text_format.MessageToString(pipeline)

with tf.gfile.Open(config_path, "wb") as f:
	f.write(config_text)
