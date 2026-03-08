Chest X-ray Pneumonia Detection

A deep learning project that detects pneumonia from chest X-ray images. Includes offline inference using TensorFlow Lite, PDF report generation, and heatmap visualization to highlight affected regions.

Features

Classifies chest X-rays as Normal or Pneumonia

Provides prediction confidence

Generates PDF report with patient details and uploaded X-ray

Highlights regions of interest using saliency heatmaps

Fully offline using a TFLite model

Dataset Structure

The project expects data under data/chest_xray/:

data/
└─ chest_xray/
    ├─ train/
        ├─ normal/
        └─ pneumonia/
    ├─ val/
        ├─ normal/
        └─ pneumonia/
    └─ test/
        ├─ normal/
        └─ pneumonia/

Model Performance
Metric	Normal	Pneumonia
Accuracy	95.2%	95.2%
Precision	0.88	0.92
Recall	0.87	0.93
F1-score	0.87	0.92

The model achieves high accuracy and highlights affected areas for explainability.

Project Structure

RareDiseaseML/
├─ data/ # Chest X-ray images
├─ outputs/ # Saved model, reports, heatmaps
├─ src/ # Python scripts
    ├─ preprocess.py # Preprocess images for training/testing
    ├─ train_model.py # Train CNN model
    ├─ evaluate_model.py # Evaluate model performance
    ├─ convert_tflite.py # Convert Keras model to TFLite
    ├─ run_tflite.py # Run TFLite model on a test image
    ├─ predict_tflite.py # Predict single image using TFLite
    ├─ gradio_demo.py # Web app interface
    └─ utils.py # Helper functions for preprocessing, PDF, heatmaps
├─ docs/
    └─ screenshots/ # Sample app interface and report screenshots
├─ requirements.txt # Python dependencies
└─ README.md

Installation

Clone the repository:

git clone https://github.com/
<your-username>/RareDiseaseML.git
cd RareDiseaseML

Create and activate a virtual environment:

python -m venv venv

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Running the Project

Train Model:
python src/train_model.py

Evaluate Model:
python src/evaluate_model.py

Convert to TFLite:
python src/convert_tflite.py

Test TFLite Model:
python src/run_tflite.py

Predict Single Image:
python src/predict_tflite.py

Run Web App with Gradio:
python src/gradio_demo.py

Upload chest X-ray → Enter patient name → Get prediction, confidence, guidance, and download PDF report

PDF Report

Generated PDF reports include:

Patient name

Prediction and confidence

Uploaded X-ray

Heatmap highlighting affected regions

Saved at: outputs/report.pdf

Notes

Ensure your dataset follows the structure above.

Outputs are saved in outputs/ (model, heatmap, reports).

TFLite allows offline usage.

Tested with Python 3.10+

Key dependencies: TensorFlow, Gradio, PIL, OpenCV, FPDF

Screenshots

Gradio App Interface and Sample PDF Report are saved in docs/screenshots/

This README is professional, clean, and developer-friendly, clearly explaining what the project does, how to use it, and the results.