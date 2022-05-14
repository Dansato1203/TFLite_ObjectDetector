import tensorflow as tf

print(tf.__version__)

converter = tf.lite.TFLiteConverter.from_saved_model('/src/train_logs',signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

with tf.gfile.GFile('/src/train_logs/ssdlite_mobiledet_colorballs.tflite', 'wb') as f:
  f.write(tflite_model)
