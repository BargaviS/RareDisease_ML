import gradio as gr
from utils import preprocess_image, predict_tflite, generate_pdf_report, get_patient_guidance
import os

MODEL_PATH = "outputs/my_model.tflite"

def predict_image(image_path, patient_name="Unknown"):
    try:
        img_array = preprocess_image(image_path)
        predicted_class, confidence = predict_tflite(MODEL_PATH, img_array)
        guidance = get_patient_guidance(predicted_class, confidence)
        report_path = generate_pdf_report(patient_name, image_path, predicted_class, confidence)
        return predicted_class, f"{confidence:.2f}", guidance, report_path
    except Exception as e:
        return f"Error: {e}", "", "", None

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="filepath", label="Chest X-ray Image"),
        gr.Textbox(label="Patient Name", placeholder="Enter patient name...")
    ],
    outputs=[
        gr.Label(label="Predicted Class"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Guidance"),
        gr.File(label="Download PDF Report")
    ],
    title="Chest X-ray Pneumonia Detection",
    description="Upload a chest X-ray image to detect pneumonia. The app provides prediction, confidence, guidance, and a PDF report."
)

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    iface.launch(debug=True)