# Chest X-ray Pneumonia Detection

A deep learning project that detects **pneumonia** from chest X-ray images.  
The system supports **offline inference using TensorFlow Lite**, generates **PDF reports**, and highlights affected areas with **heatmaps**.

---

## 🚀 Overview

Pneumonia is a serious lung infection. Early detection from chest X-rays can save lives.  
This project uses a Convolutional Neural Network (CNN) trained on chest X-ray images to classify **Normal** vs **Pneumonia**.  
It generates **confidence scores**, **visual heatmaps**, and **PDF reports** for each patient.

---

## ✨ Key Features

- Detects **Normal vs Pneumonia** from chest X-rays.
- Provides **confidence scores** for predictions.
- Generates **PDF reports** with patient name, X-ray, and prediction.
- Highlights affected regions using **heatmaps**.
- Fully offline using a **TFLite model**, lightweight and fast.
- Interactive **Gradio web app** for easy usage.

---

## 🧠 How It Works

1. Preprocess X-ray images for model input.
2. Run predictions using the trained CNN model.
3. Generate heatmaps to visualize affected areas.
4. Create PDF report containing:
   - Patient name
   - Prediction and confidence
   - Uploaded X-ray
   - Heatmap visualization
5. (Optional) Use Gradio app to upload images and get results live.

---

## 📊 Model Performance

- **Accuracy:** 95.2%
- **Precision:** 0.88 (Normal), 0.92 (Pneumonia)
- **Recall:** 0.87 (Normal), 0.93 (Pneumonia)
- **F1-score:** 0.87 (Normal), 0.92 (Pneumonia)

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/RareDiseaseML.git
cd RareDiseaseML

Create and activate a virtual environment:

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
---
▶️ How to Run

Train the model


python src/train_model.py


Evaluate the model


python src/evaluate_model.py


Convert model to TFLite


python src/convert_tflite.py


Test TFLite model on an image


python src/run_tflite.py


Predict single image using TFLite


python src/predict_tflite.py


Launch Gradio Web App


python src/gradio_demo.py


Upload a chest X-ray, enter patient name, and get:


Prediction (Normal / Pneumonia)


Confidence


Guidance
---

Downloadable PDF report

📄 PDF Report

Saved in outputs/report.pdf. Includes:

Patient name

Prediction and confidence

Uploaded X-ray

Heatmap highlighting affected regions
---
⚠️ Notes

Follow the dataset structure strictly.

All outputs (model, heatmaps, reports) are saved in outputs/.

TFLite model allows offline inference.

Gradio app provides a simple interface for non-technical users.
---
🛠️ Technologies Used

Python

TensorFlow / TensorFlow Lite

OpenCV

PIL (Pillow)

FPDF

Gradio
---
👩‍💻 Author

Bargavi S
Aspiring AI Engineer

GitHub: https://github.com/BargaviS

LinkedIn: https://linkedin.com/in/bargavis
---
⭐ Acknowledgment

If you found this project useful, consider giving it a star ⭐ on GitHub.



