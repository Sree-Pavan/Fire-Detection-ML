import tensorflow as tf

model = tf.keras.models.load_model('InceptionV3.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("InceptionV3.tflite", "wb").write(tflite_model)