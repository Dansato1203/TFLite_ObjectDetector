#! /bin/bash

cd /src/models/research
tflite_convert \
  --output_file="/src/train_logs/tflite/ssdlite_mobiledet_colorballs.tflite" \
  --graph_def_file="/src/train_logs/tflite_graph.pb" \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays="normalized_input_image_tensor" \
  --output_arrays="TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3" \
  --mean_values=128 \
  --std_dev_values=128 \
  --input_shapes=1,320,320,3 \
  --allow_custom_ops

edgetpu_compiler -s /src/train_logs/tflite/ssdlite_mobiledet_colorballs.tflite
