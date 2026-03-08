import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import glob

tflite_model_path = Path("outputs/my_model.tflite")

test_folder = Path("data/chest_xray/test/pneumonia")

if len(sys.argv) > 1:
    sample_image_path = Path(sys.argv[1])
else:
    
    images = glob.glob(str(test_folder / "*.jpg")) + glob.glob(str(test_folder / "*.jpeg")) + glob.glob(str(test_folder / "*.png"))
    if not images:
        raise FileNotFoundError(f"No images found in {test_folder}")
    sample_image_path = Path(images[0])

if not sample_image_path.exists():
    raise FileNotFoundError(f"Test image not found: {sample_image_path}")

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open(sample_image_path).convert("RGB")
img = img.resize((224, 224))  
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0) 

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

classes = ["normal", "pneumonia"]
pred_class = classes[np.argmax(output)]
pred_conf = float(np.max(output))

print(f"Image: {sample_image_path.name}")
print(f"Predicted class: {pred_class}")
print(f"Confidence: {pred_conf:.2f}")