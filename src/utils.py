import numpy as np
from PIL import Image
import tensorflow as tf
from fpdf import FPDF
import os

def preprocess_image(image_path, target_size=(224,224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_tflite(model_path, image_array):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, image_array.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)[0]

    if prediction.shape[0] == 1:  
        confidence = float(prediction[0])
        predicted_class = "pneumonia" if confidence >= 0.5 else "normal"
    elif prediction.shape[0] == 2:  
        predicted_class = "pneumonia" if prediction[1] > prediction[0] else "normal"
        confidence = float(max(prediction))
    else:
        raise ValueError("Unexpected model output shape:", prediction.shape)
    
    return predicted_class, confidence

def get_patient_guidance(predicted_class, confidence):
    if predicted_class == "pneumonia" and confidence >= 0.7:
        return "High probability of pneumonia. Consult a doctor immediately."
    elif predicted_class == "pneumonia":
        return "Moderate probability of pneumonia. Recommend a medical check-up."
    else:
        return "Normal. Maintain healthy lifestyle."

def generate_pdf_report(patient_name, image_path, predicted_class, confidence, output_path="outputs/report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Patient Report", ln=True, align="C")
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Patient: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Prediction: {predicted_class} (Confidence: {confidence:.2f})", ln=True)
    pdf.ln(5)
    pdf.image(image_path, x=60, w=90)
    pdf.output(output_path)
    return output_path