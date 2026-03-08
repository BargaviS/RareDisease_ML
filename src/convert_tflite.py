import tensorflow as tf
import os

model_path = 'outputs/my_model.keras'
model = tf.keras.models.load_model(model_path)

os.makedirs('outputs', exist_ok=True)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  
tflite_model = converter.convert()

tflite_model_path = 'outputs/my_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Model successfully converted to TFLite and saved at {tflite_model_path}")