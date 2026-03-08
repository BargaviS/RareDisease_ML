# Chest X-ray Pneumonia Detection

A deep learning project that detects pneumonia from chest X-ray images. The system supports offline inference using TensorFlow Lite, generates PDF reports, and highlights affected areas with heatmaps.

---

## 🚀 Overview

Pneumonia is a serious lung infection. Early detection from chest X-rays can save lives. This project uses a Convolutional Neural Network (CNN) trained on chest X-ray images to classify **Normal vs Pneumonia**.

It provides:

- Confidence scores for predictions
- Heatmaps to visualize affected areas
- PDF reports containing patient name, X-ray, prediction, and heatmaps
- Optional Gradio web app for interactive use

---

## ✨ Key Features

- Detects **Normal vs Pneumonia** from chest X-rays
- Provides **confidence scores** for predictions
- Generates **PDF reports** with patient name, X-ray, and prediction
- Highlights affected regions using **heatmaps**
- Fully **offline using a TFLite model** (lightweight and fast)
- Interactive **Gradio web app** for easy usage

---

## 🧠 How It Works

1. Preprocess X-ray images for model input  
2. Run predictions using the trained CNN model  
3. Generate heatmaps to visualize affected areas  
4. Create PDF reports containing:
   - Patient name
   - Prediction and confidence
   - Uploaded X-ray
   - Heatmap visualization

Optionally, use the **Gradio web app** to upload images and get live results.

---

## 📊 Model Performance

| Metric | Value |
|------|------|
| Accuracy | 95.2% |
| Precision | 0.88 (Normal), 0.92 (Pneumonia) |
| Recall | 0.87 (Normal), 0.93 (Pneumonia) |
| F1-score | 0.87 (Normal), 0.92 (Pneumonia) |

---


Usage:

Upload a chest X-ray, enter patient name, and get:

- Prediction (Normal / Pneumonia)
- Confidence score
- Patient guidance
- Downloadable PDF report

---

## 📄 PDF Report

Saved in `outputs/report.pdf`.

The report includes:

- Patient name
- Prediction and confidence
- Uploaded X-ray
- Heatmap highlighting affected regions

---

## ⚠️ Notes

- Follow the dataset structure strictly.
- All outputs (model, heatmaps, reports) are saved in `outputs/`.
- TFLite model allows **offline inference**.
- Gradio app provides a **simple interface for non-technical users**.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / TensorFlow Lite
- OpenCV
- PIL (Pillow)
- FPDF
- Gradio

---

## 👩‍💻 Author

**Bargavi S**  
Aspiring AI Engineer  

GitHub: https://github.com/BargaviS  
LinkedIn: https://linkedin.com/in/bargavis

---

## ⭐ Acknowledgment

If you found this project useful, consider giving it a **star ⭐ on GitHub**.
