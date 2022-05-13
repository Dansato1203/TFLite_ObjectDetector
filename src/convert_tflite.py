import tensorflow as tf

print(tf.__version__)

#converter = tf.lite.TFLiteConverter.from_saved_model('/src/exported_graphs/saved_model',signature_keys=['serving_default'])
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.experimental_new_converter = True
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

#tflite_model = converter.convert()

#with tf.io.gfile.GFile('model.tflite', 'wb') as f:
#  f.write(tflite_model)

def representative_dataset_gen():
  for i in range(0,5):
    x,y=train_generator.next()
    image=x[i:i+1]
    yield [image]

converter = tf.lite.TFLiteConverter.from_saved_model('/src/exported_graphs/saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_dataset_gen()
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.uint8  # or tf.uint8
#converter.inference_output_type = tf.uint8  # or tf.uint8
tflite_quant_model = converter.convert()
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
    f.write(tflite_model)
