import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random

tflite_model_path = 'outputs/my_model.tflite'
test_dir = 'data/chest_xray/test/pneumonia' 
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test folder not found: {test_dir}")

images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not images:
    raise FileNotFoundError(f"No images found in {test_dir}")

sample_image_path = os.path.join(test_dir, random.choice(images))

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open(sample_image_path).convert("RGB")
img = img.resize((224, 224))  
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

class_name = "pneumonia" if pred >= 0.5 else "normal"
confidence = pred if pred >= 0.5 else 1 - pred

print(f"Image: {os.path.basename(sample_image_path)}")
print(f"Predicted class: {class_name}")
print(f"Confidence: {confidence:.2f}")