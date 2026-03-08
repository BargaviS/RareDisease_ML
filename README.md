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

| Metric    | Value |
|-----------|-------|
| Accuracy  | 95.2% |
| Precision | 0.88 (Normal), 0.92 (Pneumonia) |
| Recall    | 0.87 (Normal), 0.93 (Pneumonia) |
| F1-score  | 0.87 (Normal), 0.92 (Pneumonia) |

---

## Project Structure

---

## Model Performance

| Metric    | Value |
|-----------|-------|
| Accuracy  | 95.2% |
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

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Running the Project

Train Model

python src/train_model.py

Evaluate Model

python src/evaluate_model.py

Convert to TFLite

python src/convert_tflite.py

Test TFLite Model

python src/run_tflite.py

Predict Single Image

python src/predict_tflite.py

Run Gradio Web App

python src/gradio_demo.py

Upload chest X-ray → Enter patient name → Get prediction, confidence, guidance, and download PDF report.

PDF Report

Saved in outputs/report.pdf
Includes:

Patient name

Prediction and confidence

Uploaded X-ray

Heatmap highlighting affected regions

Notes

Ensure the dataset path follows the structure above.

Outputs are saved in outputs/ (model, heatmap, report).

TFLite model allows offline usage.

Recommended Python version: 3.10+

Tested with: TensorFlow, Gradio, PIL, OpenCV, FPDF

This README is professional, clear, and developer-friendly, showing exactly what the project solves and how to run it.


---

If you want, I can **also make a ready-to-use `requirements.txt`** with **exact versions of TensorFlow, Gradio, PIL, OpenCV, FPDF**, so anyone can just clone your repo and run the project **without errors**.  

Do you want me to create that next?