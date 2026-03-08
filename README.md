# Chest X-ray Pneumonia Detection

A deep learning project that detects **pneumonia** from chest X-ray images.  
Includes **offline inference using TensorFlow Lite**, **PDF report generation**, and **heatmap visualization**.

---

## Features

- Detects **Normal** vs **Pneumonia** from chest X-rays.
- Provides **confidence score** for predictions.
- Generates **PDF report** with patient details and X-ray.
- **Saliency heatmaps** highlight affected regions.
- Fully offline using **TFLite model**.

---

## Dataset Structure

The project expects data under `data/chest_xray/`:
data/
└─ chest_xray/
├─ train/
│ ├─ normal/
│ └─ pneumonia/
├─ val/
│ ├─ normal/
│ └─ pneumonia/
└─ test/
├─ normal/
└─ pneumonia/


---

## Model Performance

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 95.2%  |
| Precision | 0.88 (Normal), 0.92 (Pneumonia) |
| Recall    | 0.87 (Normal), 0.93 (Pneumonia) |
| F1-score  | 0.87 (Normal), 0.92 (Pneumonia) |

---

## Project Structure


RareDiseaseML/
│
├─ data/ # Chest X-ray images
├─ outputs/ # Model, reports, heatmaps
├─ src/ # Python scripts
│ ├─ preprocess.py
│ ├─ train_model.py
│ ├─ evaluate_model.py
│ ├─ convert_tflite.py
│ ├─ run_tflite.py
│ ├─ predict_tflite.py
│ ├─ gradio_demo.py
│ └─ utils.py
├─ docs/
│ └─ screenshots/ # App interface and report examples
├─ requirements.txt
└─ README.md


---

## Installation

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
Running the Project
1. Train Model
python src/train_model.py
2. Evaluate Model
python src/evaluate_model.py
3. Convert to TFLite
python src/convert_tflite.py
4. Test TFLite Model
python src/run_tflite.py
5. Predict Single Image
python src/predict_tflite.py
6. Run Gradio App
python src/gradio_demo.py

Upload chest X-ray

Enter patient name

Get prediction, confidence, guidance, and download PDF report

Scripts Overview
Script	Purpose
preprocess.py	Preprocess images for training/testing
train_model.py	Train CNN on dataset
evaluate_model.py	Evaluate model performance
convert_tflite.py	Convert Keras model to TFLite
run_tflite.py	Run TFLite model on a test image
predict_tflite.py	Predict class using TFLite
gradio_demo.py	Web app interface using Gradio
utils.py	Helper functions for preprocessing, predictions, PDF, heatmaps
PDF Report

Saved in outputs/report.pdf

Includes:

Patient name

Prediction and confidence

Uploaded X-ray

Heatmap highlighting affected regions

Screenshots

Gradio App Interface


Sample PDF Report


Notes

Make sure the dataset path follows the structure above.

Outputs are saved in outputs/ (model, heatmap, report).

TFLite model allows offline usage.


---

This README is **clean, developer-friendly, and professional**. It avoids mentioning hackathons or AI text style.  

---

If you want, I can **also create a ready-to-use `requirements.txt`** with **exact versions of TensorFlow, Gradio, PIL, OpenCV, FPDF**, so anyone can just run the project without errors.  

Do you want me to do that next?